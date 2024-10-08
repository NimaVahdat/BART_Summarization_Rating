import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

from utils import three_sentence_summary


class DataProcessor:
    def __init__(
        self,
        dataset_name: str = None,
        category: str = "raw_review_All_Beauty",
        tokenizer: AutoTokenizer = None,
        split: float = 0.2,
    ) -> None:
        # Load the full dataset
        full_data = load_dataset(
            dataset_name, category, split="full", trust_remote_code=True
        )

        # Filter out examples with very short titles to improve the quality of summaries
        full_data = full_data.filter(lambda x: len(x["title"].split()) > 3)

        # Split the dataset into training and validation sets
        full_data_split = full_data.train_test_split(test_size=split, seed=1234)
        self.dataset_train = full_data_split["train"]
        self.dataset_val = full_data_split["test"]

        # Initialize the tokenizer
        self.tokenizer = tokenizer

    def _preprocess_function(self, examples):
        # Tokenize the text inputs
        inputs = examples["text"]
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)

        # Tokenize the titles to use as labels for the model
        labels = self.tokenizer(examples["title"], max_length=30, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        # Convert ratings to integer and add to model inputs
        model_inputs["ratings"] = list(map(int, examples["rating"]))

        return model_inputs

    def _remove_unnecessary_columns(self, dataset):
        # Retain only the necessary columns for model training
        necessary_columns = ["input_ids", "attention_mask", "labels", "ratings"]
        for column in dataset.column_names:
            if column not in necessary_columns:
                dataset = dataset.remove_columns(column)
        return dataset

    def __call__(self, mode: str = None):
        # Preprocess and return the dataset based on the specified mode
        if mode == "train":
            return self.dataset_train.map(self._preprocess_function, batched=True)
        elif mode in ["validation", "eval", "val"]:
            return self.dataset_val.map(self._preprocess_function, batched=True)
        else:
            train_dataset = self.dataset_train.map(
                self._preprocess_function, batched=True
            )
            val_dataset = self.dataset_val.map(self._preprocess_function, batched=True)
            train_dataset = self._remove_unnecessary_columns(train_dataset)
            val_dataset = self._remove_unnecessary_columns(val_dataset)
            return train_dataset, val_dataset

    def get_baseline(self):
        # Compute baseline ROUGE score by summarizing validation texts and comparing to titles.
        rouge_score = evaluate.load("rouge")
        summaries = [three_sentence_summary(text) for text in self.dataset_val["text"]]
        score = rouge_score.compute(
            predictions=summaries, references=self.dataset_val["title"]
        )
        return score
