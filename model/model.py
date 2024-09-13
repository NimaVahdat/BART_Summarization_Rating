import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartConfig
from model.modeling_outputs import ModelOutputForReviewSummarizationAndRating
from transformers.utils import replace_return_docstrings
from transformers.models.bart.modeling_bart import shift_tokens_right


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        innner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, innner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(innner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)

        return hidden_states


class BartForReviewSummarizationAndRating(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        # Classification head for rating prediction (5 classes: 1 to 5)
        self.classifier = ClassificationHead(
            config.d_model,
            config.d_model,
            5,
            config.classifier_dropout,
        )
        # Optional loss weighting for multitask learning
        self.loss_weight_summarization = 1.0
        self.loss_weight_classification = 1.0

    def freeze_summarization_head(self):
        # Freeze the BART model (summarization head)
        for param in self.model.parameters():
            param.requires_grad = False
        self.loss_weight_summarization = 0

    def freeze_classification_head(self):
        # Freeze the classification head
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.loss_weight_classification = 0

    def unfreeze_summarization_head(self):
        # Unfreeze the BART model (summarization head)
        for param in self.model.parameters():
            param.requires_grad = True
        self.loss_weight_summarization = 1.0

    def unfreeze_classification_head(self):
        # Unfreeze the classification head
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.loss_weight_classification = 1.0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        ratings=None,
        return_dict=True,
        **kwargs,
    ):

        # Perform the forward pass for summarization (title generation)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        if input_ids is None:
            return outputs  # Handling self.generate() method

        hidden_states = outputs.decoder_hidden_states[-1]

        # Create an EOS mask for the decoder's input IDs
        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        eos_indices = eos_mask.nonzero(as_tuple=False)

        eos_indices = eos_mask.sum(dim=1) - 1

        batch_indices = torch.arange(hidden_states.size(0))
        sentence_representation = hidden_states[batch_indices, eos_indices, :]

        rating_logits = self.classifier(sentence_representation)

        # Calculate the loss for rating prediction if ratings are provided
        rating_loss = None
        if ratings is not None:
            ratings = ratings.long() - 1  # Zero-indexing: rating range [1-5] -> [0-4]
            loss_fct = nn.CrossEntropyLoss()
            rating_loss = loss_fct(rating_logits, ratings)

        # Combine the summarization loss and the rating loss, with optional weighting
        combined_loss = outputs.loss
        if rating_loss is not None:
            combined_loss = (
                self.loss_weight_classification * rating_loss
                + self.loss_weight_summarization * outputs.loss
            )

        if return_dict:
            return ModelOutputForReviewSummarizationAndRating(
                loss=combined_loss,
                logits=outputs.logits,
                rating_loss=rating_loss,
                summarization_loss=outputs.loss,
                rating_logits=rating_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        else:
            return combined_loss, rating_logits
