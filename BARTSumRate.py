from data import DataProcessor
from utils import postprocess_text

from model import BartForReviewSummarizationAndRating
from transformers import AutoTokenizer, GenerationConfig, DataCollatorForSeq2Seq

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from transformers import get_scheduler

from tqdm.auto import tqdm
import evaluate
import numpy as np


class BART_Sum_Rate:
    def __init__(self, data_info: dict = None, training_info: dict = None) -> None:
        self.training_info = training_info
        self.data_info = data_info

        # Load model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(data_info["tokenizer_info"])

        self._define_training_components()

        # Setup and evaluation metrics
        self.save_path = training_info["save_path"]
        self.rouge_score = evaluate.load("rouge")
        self.accuracy_metric = evaluate.load("accuracy")

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=training_info["log_dir"])

    def _define_training_components(self):
        # Check if summarization weights are provided, else initialize from pre-trained BART
        if not self.training_info["load_checkpoint"]:
            self.head = "summarization"

            self.model = BartForReviewSummarizationAndRating.from_pretrained(
                self.training_info["config"]
            )
            self.model.freeze_classification_head()

            self._set_or_reset_training_factors(head="summarization")

            # Set total epochs for the full training cycle
            self.num_epochs = self.training_info["num_epochs"]
        else:
            self.head = "classification"

            # Switch to classification head
            self._switch_heads_to_classification()

            # Adjust total epochs for classification training
            self.num_epochs = (
                self.training_info["num_epochs"] - self.training_info["mid_epoch"]
            )

    def _set_or_reset_training_factors(self, head="summarization"):
        """Set or reset training factors based on the head being trained."""
        self.batch_size = self.training_info[head]["batch_size"]
        self.train_dataloader, self.eval_dataloader = self._get_data_loader(
            self.data_info, self.batch_size
        )
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.training_info[head]["lr"],
            weight_decay=self.training_info[head]["weight_decay"],
        )

        # Prepare model and optimizer for distributed training
        self.accelerator = Accelerator()
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
            )
        )

        num_epochs = (
            self.training_info["num_epochs"]
            if head == "summarization"
            else self.training_info["num_epochs"] - self.training_info["mid_epoch"]
        )
        self.scheduler = self._get_scheduler(num_epochs=num_epochs)

        self.training_info.update(
            {
                "best_val_loss": float("inf"),
                "best_val_rouge": 0,
                "best_val_accuracy": 0,
            }
        )

    def _get_dataset(self, dataset_name: str, category: str, tokenizer: str):
        data_processor = DataProcessor(dataset_name, category, tokenizer)
        print("Base dataset statistics:", data_processor.get_baseline())
        return data_processor()

    def _get_data_loader(self, data_info: dict, batch_size: int = 8):
        dataset_name = data_info["dataset_name"]
        category = data_info["category"]

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        train_dataset, eval_dataset = self._get_dataset(
            dataset_name, category, self.tokenizer
        )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
        )

        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=batch_size
        )

        return train_dataloader, eval_dataloader

    def _get_scheduler(self, num_epochs):
        num_update_steps_per_epoch = len(self.train_dataloader)
        self.num_training_steps = num_epochs * num_update_steps_per_epoch

        return get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_rating_loss = 0.0
            correct_ratings = 0
            total_samples = 0

            if epoch > self.training_info["mid_epoch"] and self.head == "summarization":
                self._switch_heads_to_classification()
                self.head = "classification"

            progress_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Epoch {epoch+1} - training {self.head} head",
                leave=True,
            )

            for step, batch in enumerate(self.train_dataloader):
                output = self.model(**batch)
                loss = output["loss"]
                rating_loss = output["rating_loss"]

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                if rating_loss is not None:
                    epoch_rating_loss += rating_loss.item()

                # Track rating accuracy
                rating_logits = output["rating_logits"]
                correct_ratings += (
                    (rating_logits.argmax(dim=-1) + 1 == batch["ratings"]).sum().item()
                )
                total_samples += batch["ratings"].size(0)

                progress_bar.update(1)

            progress_bar.close()

            avg_train_loss = epoch_loss / len(self.train_dataloader)
            avg_rating_loss = epoch_rating_loss / len(self.train_dataloader)
            rating_accuracy = correct_ratings / total_samples

            # Log training loss, rating loss, and accuracy
            self.writer.add_scalar(f"Loss/Train_{self.head}", avg_train_loss, epoch)
            self.writer.add_scalar(
                f"Loss/Rating_Train_{self.head}", avg_rating_loss, epoch
            )
            self.writer.add_scalar(
                f"Accuracy/Train_{self.head}", rating_accuracy, epoch
            )

            print(
                f"Epoch {epoch+1} complete. Avg Training Loss: {avg_train_loss:.4f}, Rating Loss: {avg_rating_loss:.4f}, Rating Accuracy: {rating_accuracy:.4f}"
            )

            # Validation
            val_loss, val_rouge, val_rating_loss, val_rating_accuracy = self.eval(epoch)

            # Log validation loss, ROUGE scores, and accuracy
            self.writer.add_scalar(f"Loss/Validation_{self.head}", val_loss, epoch)
            self.writer.add_scalar(
                f"Loss/Rating_Validation_{self.head}", val_rating_loss, epoch
            )
            self.writer.add_scalar(
                f"Accuracy/Validation_{self.head}", val_rating_accuracy, epoch
            )

            print(
                f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Rating Loss: {val_rating_loss:.4f}, Rating Accuracy: {val_rating_accuracy:.4f}"
            )
            print(f"Epoch {epoch+1} ROUGE: {val_rouge}")

            # Save the model if validation loss improves
            self.save_model(epoch, val_loss, val_rouge["rouge2"], val_rating_accuracy)

        self.writer.close()

    def _switch_heads_to_classification(self):
        """Switch training to the classification head."""
        self.model = BartForReviewSummarizationAndRating.from_pretrained(
            self.training_info["summarization_weights_path"]
        )
        self.model.freeze_summarization_head()
        self.model.unfreeze_classification_head()
        self._set_or_reset_training_factors(head="classification")

    def eval(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        val_rating_loss = 0.0
        correct_ratings = 0
        total_samples = 0
        progress_bar = tqdm(
            range(len(self.eval_dataloader)),
            desc=f"Evaluating Epoch {epoch+1}",
            leave=True,
        )

        all_decoded_preds, all_decoded_labels = [], []

        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                output = self.model(**batch)
                loss = output["loss"]
                rating_loss = output["rating_loss"]
                rating_logits = output["rating_logits"]

                val_loss += loss.item()
                if rating_loss is not None:
                    val_rating_loss += rating_loss.item()

                # Track rating accuracy
                correct_ratings += (
                    (rating_logits.argmax(dim=-1) + 1 == batch["ratings"]).sum().item()
                )
                total_samples += batch["ratings"].size(0)

                # ####### Generate predictions for ROUGE scoring (Faster but not exact) ##########
                # generated_tokens = torch.argmax(output["logits"], dim=-1)

                # # Pad and gather across processes if needed
                # generated_tokens = self.accelerator.pad_across_processes(
                #     generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                # )
                # labels = self.accelerator.pad_across_processes(
                #     batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id
                # )

                # generated_tokens = (
                #     self.accelerator.gather(generated_tokens).cpu().numpy()
                # )
                # labels = self.accelerator.gather(labels).cpu().numpy()

                # labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

                # # Decode generated tokens and labels
                # decoded_preds = self.tokenizer.batch_decode(
                #     generated_tokens, skip_special_tokens=True
                # )
                # decoded_labels = self.tokenizer.batch_decode(
                #     labels, skip_special_tokens=True
                # )

                # all_decoded_preds.extend(decoded_preds)
                # all_decoded_labels.extend(decoded_labels)
                # ################################################################################

                ####### Generate predictions for ROUGE scoring (Slower but exact) ##########
                generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=30,
                )

                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = self.accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id
                )

                generated_tokens = (
                    self.accelerator.gather(generated_tokens).cpu().numpy()
                )
                labels = self.accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                all_decoded_preds.extend(decoded_preds)
                all_decoded_labels.extend(decoded_labels)
                ################################################################################

                progress_bar.update(1)

        progress_bar.close()

        all_decoded_preds, all_decoded_labels = postprocess_text(
            all_decoded_preds, all_decoded_labels
        )

        # Compute ROUGE score
        val_rouge = self.rouge_score.compute(
            predictions=all_decoded_preds, references=all_decoded_labels
        )

        avg_val_loss = val_loss / len(self.eval_dataloader)
        avg_val_rating_loss = val_rating_loss / len(self.eval_dataloader)
        val_rating_accuracy = correct_ratings / total_samples

        return avg_val_loss, val_rouge, avg_val_rating_loss, val_rating_accuracy

    def save_model(
        self, epoch, validation_loss, validation_rouge=None, validation_accuracy=None
    ):
        """
        Save the model if the validation loss improves or based on additional metrics like ROUGE or rating accuracy.

        Args:
            epoch (int): Current epoch number.
            validation_loss (float): Current validation loss.
            validation_rouge (float, optional): ROUGE score from summarization evaluation.
            validation_accuracy (float, optional): Rating prediction accuracy.
        """
        # Save based on validation loss improvement
        if validation_loss < self.training_info["best_val_loss"]:
            self.training_info["best_val_loss"] = validation_loss
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                f"{self.save_path}/best_loss_model_{self.head}"
            )
            print(
                f"Model saved at epoch {epoch+1} with improved validation loss: {validation_loss:.4f}"
            )

        # Optionally, save based on ROUGE score improvement
        if validation_rouge is not None and validation_rouge > self.training_info.get(
            "best_val_rouge", 0
        ):
            self.training_info["best_val_rouge"] = validation_rouge
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                f"{self.save_path}/best_rouge_model_{self.head}"
            )
            print(
                f"Model saved at epoch {epoch+1} with improved ROUGE score: {validation_rouge:.4f}"
            )

        # Optionally, save based on rating accuracy improvement
        if (
            validation_accuracy is not None
            and validation_accuracy > self.training_info.get("best_val_accuracy", 0)
        ):
            self.training_info["best_val_accuracy"] = validation_accuracy
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                f"{self.save_path}/best_accuracy_model_{self.head}"
            )
            print(
                f"Model saved at epoch {epoch+1} with improved rating accuracy: {validation_accuracy:.4f}"
            )
