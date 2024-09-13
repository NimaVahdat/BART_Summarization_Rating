import yaml
from BARTSumRate import BART_Sum_Rate


def load_config(config_file):
    """
    Load configuration settings from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Load configuration from YAML file
    data_info = load_config("config/data_info.yaml")
    training_info = load_config("config/training_info.yaml")

    # Initialize the SegmentationTrainer with loaded configuration and datasets
    trainer = BART_Sum_Rate(data_info=data_info, training_info=training_info)

    # Start training the model
    trainer.train()
