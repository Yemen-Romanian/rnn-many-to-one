import tomllib
from torch.utils.data import DataLoader

from factories import build_model, build_optimizer
from dataset import SyntheticSequenceRegressionDataset
from train import Trainer

def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    dataset_config = config["dataset"]
    train_dataset = SyntheticSequenceRegressionDataset(
        num_samples=dataset_config["train_size"],
        sequence_length=dataset_config["sequence_length"],
        num_features=dataset_config["num_features"]
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["dataloader"]["batch_size"], 
        shuffle=True
    )

    test_dataset = SyntheticSequenceRegressionDataset(
        num_samples=dataset_config["test_size"],
        sequence_length=dataset_config["sequence_length"],
        num_features=dataset_config["num_features"]
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["dataloader"]["batch_size"], 
        shuffle=False
    )

    model = build_model(config)
    optimizer = build_optimizer(
            model=model, 
            config=config
        )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        epochs=config["trainer"]["epochs"]
    )

    print("Configuration Loaded:")

    trainer.train()


if __name__ == "__main__":
    main()