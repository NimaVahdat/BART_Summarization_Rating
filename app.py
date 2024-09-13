import torch
from transformers import AutoTokenizer
from model import BartForReviewSummarizationAndRating
from flask import Flask, render_template, request, jsonify
import gradio as gr

# Create Flask app
app = Flask(__name__)


class InferenceBARTSumRate:
    def __init__(self, model_path, tokenizer_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load the model and put it on the appropriate device
        self.model = BartForReviewSummarizationAndRating.from_pretrained(model_path)
        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

    def summarize_and_predict_rating(self, review_text):
        # Tokenize the input review text
        inputs = self.tokenizer(
            review_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move inputs to the same device as the model
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Perform forward pass for summarization and rating prediction
        with torch.no_grad():
            rating_logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )[1]
            outputs = self.model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=150
            )

        # Decode the generated summary
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Predict the rating
        predicted_rating = (
            torch.argmax(rating_logits, dim=-1).item() + 1
        )  # Adding 1 as ratings are 1-5

        return summary, predicted_rating


def generate_summary_and_rating(review_text):
    # Paths to model and tokenizer
    model_path = "saved_models/bart_sum_rate/best_rouge_model_classification"  # Replace with your trained model path
    tokenizer_path = "facebook/bart-base"

    # Instantiate the inference class
    inferencer = InferenceBARTSumRate(
        model_path=model_path, tokenizer_path=tokenizer_path
    )

    # Perform inference
    summary, rating = inferencer.summarize_and_predict_rating(review_text)

    return summary, rating


def display_stars(rating):
    """
    Generate star ratings based on the predicted rating.
    """
    stars_html = '<div class="stars">'
    for i in range(1, 6):  # 5 stars
        if i <= rating:
            stars_html += (
                '<span align="center" style="color: yellow; font-size: 48px;">★</span>'
            )
        else:
            stars_html += (
                '<span align="center" style="color: gray; font-size: 48px;">★</span>'
            )
    stars_html += "</div>"
    return stars_html


# Gradio Interface function
def process_input(review_text):
    summary, rating = generate_summary_and_rating(review_text)
    stars_html = display_stars(rating)
    return summary, stars_html


# Define Gradio interface
gradio_app = gr.Blocks()

with gradio_app:
    gr.Markdown(
        "# **<p align='center'>Modified and Fine-Tuned BART for Multitask Amazon Review Summarization and Rating Analysis</p>**"
    )
    gr.Markdown("<p align='center'>By Nima Vahdat</p>")
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1.5):
                review_input = gr.Textbox(label="Enter Review Text")
    with gr.Group():
        with gr.Column(scale=1.5):
            summary_output = gr.Textbox(label="Summary")
        stars_output = gr.HTML(label="Predicted Rating (Stars)")

    # Group for the run button
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                start_run = gr.Button("Get the Summary and Rating!")

    # When user submits, it processes input and shows summary and star rating
    start_run.click(
        process_input, inputs=[review_input], outputs=[summary_output, stars_output]
    )


# Flask route to serve the Gradio app
@app.route("/gradio")
def gradio_interface():
    return gradio_app.launch(
        share=False
    )  # Launch Gradio within Flask without sharing externally


# Flask main page
@app.route("/")
def home():
    return render_template(
        "index.html"
    )  # Assuming you have an HTML template for the main page


# Flask route for cURL requests
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Expecting JSON input
    if "review_text" not in data:
        return jsonify({"error": "Missing review_text"}), 400

    review_text = data["review_text"]
    summary, rating = generate_summary_and_rating(review_text)

    return jsonify({"summary": summary, "rating": rating})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
