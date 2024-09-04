import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartConfig


class BartForReviewSummarizationAndRating(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        # Classification head for rating prediction (5 classes: 1 to 5)
        self.classifier = nn.Linear(config.d_model, 5)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        ratings=None,
    ):
        # Perform the forward pass for summarization (title generation)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        # Use the encoder's final hidden state for rating prediction
        encoder_hidden_states = outputs.encoder_last_hidden_state
        sentence_representation = encoder_hidden_states[
            :, 0, :
        ]  # Use the first token's representation

        # Apply classifier to predict the rating
        rating_logits = self.classifier(
            sentence_representation
        )  # Shape: (batch_size, 5)

        # Calculate the loss for rating prediction if ratings are provided
        rating_loss = None
        if ratings is not None:
            # Adjust ratings to be zero-indexed and ensure they are LongTensors
            ratings = ratings.long() - 1
            loss_fct = nn.CrossEntropyLoss()
            rating_loss = loss_fct(rating_logits, ratings)

        # Combine the summarization loss and the rating loss
        combined_loss = (
            outputs.loss + rating_loss if rating_loss is not None else outputs.loss
        )

        return {
            "loss": combined_loss,  # Total loss combining both summarization and rating losses
            "logits": rating_logits,  # Rating prediction logits
            "summarization_loss": outputs.loss,  # Loss from the summarization task
            "rating_loss": rating_loss,  # Loss from the rating prediction task (if available)
        }


# Example usage:
# config = BartConfig.from_pretrained("facebook/bart-base")
# model = BartForReviewSummarizationAndRating(config)
# outputs = model(input_ids, attention_mask, decoder_input_ids, labels, ratings)
