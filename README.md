# Review Summarization and Rating with BART

## Overview
This project leverages a fine-tuned BART model to perform multitask learning for Amazon review summarization and rating prediction. The model is designed to generate concise summaries of review texts and predict ratings based on the provided content. To enhance training efficiency, Hugging Face's **Accelerate** is used for distributed training, enabling multi-GPU support and mixed-precision training.

### Project Components
- **Model Training**: Fine-tuning BART for multitask learning using an Amazon review dataset. Distributed training is handled by Hugging Face's **Accelerate** to scale across multiple GPUs.
- **Inference API**: A Flask-based API integrated with a Gradio interface for easy review submission and result display.
- **Web Interface**: A user-friendly web interface to interact with the Gradio app embedded in a Flask web page.

### Features
- **Summarization**: Generates a concise summary of the review text.
- **Rating Prediction**: Predicts the rating (1-5 stars) based on the review content.
- **Web Interface**: Integrated Gradio app for real-time interaction.
- **API**: Flask API endpoint for programmatic access.
- **Distributed Training**: Hugging Face's **Accelerate** is used to efficiently train the model on multiple GPUs with mixed precision.


### Gradio Interface Demo

Here's a demonstration of the Gradio app in action:

![Gradio Demo](https://github.com/NimaVahdat/BART-Summarization-Rating/blob/main/Demo.gif)

## Getting Started

### Training the Model

1. **Get Dataset**: Please review [this](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and then update your data configuration file.

2. **Configuration Files**: Create and update your configuration files for data and training settings. Example files can be found in the `config/` directory.

3. **Run Training Script**:
   ```bash
   python train.py
   ```
   This script will use the configurations specified in your YAML files to train the BART model and save the best-performing model.

### Running the Application

1. **Start the Flask Server**:
   ```bash
   python app.py
   ```
   This command will start the Flask server, which includes the Gradio interface for interacting with your model.

2. **Access the Web Interface**:
   Open your web browser and navigate to `http://localhost:5000` to view the Gradio interface embedded within the Flask application.


### API Usage

You can also interact with the model via the Flask API. Here is an example of how to use `cURL` to get a summary and rating:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"review_text": "Your review text here"}'
```

**Response Example:**

```json
{
  "summary": "Generated summary of the review text.",
  "rating": 4
}
```

## Files

- `train.py`: Script to train the BART model with specified configurations.
- `app.py`: Flask application integrating the Gradio interface.
- `model.py`: Contains the BART model definition for review summarization and rating.
- `templates/index.html`: HTML template for embedding the Gradio interface.
- `config/data_info.yaml`: Configuration file for data-related settings.
- `config/training_info.yaml`: Configuration file for training-related settings.

## Dependencies

- `transformers`: For model and tokenizer.
- `torch`: PyTorch library for deep learning.
- `flask`: Web framework for building the API.
- `gradio`: Interface for interactive model deployment.
- `pyyaml`: For reading configuration files.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
