import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.config.load_configs import load_config, save_config
from src.data.load_data import MedMNISTDataModule
from src.evaluation.plots import save_confusion_matrix
from src.model.model_factory import build_model
from src.training.trainer import Trainer
from src.utils.set_seed import set_seed


def get_device() -> torch.device:
    """
    Return the best available device for training or inference.

    The device selection prefers CUDA when available, then Apple's MPS
    backend, and finally falls back to CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(config_path: str):
    """
    Run the end-to-end training and evaluation pipeline.

    This function loads the experiment configuration, sets the random seed,
    initializes the data module, builds the model, trains it, reloads the
    best checkpoint based on validation performance, evaluates on the test
    split, and saves run artifacts to disk.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file that defines dataset, model,
        training, and artifact settings.

    Returns
    -------
    None
        This function does not return a value. It saves artifacts to disk
        and prints training and evaluation results to stdout.
    """
    # load in config, set seed and get accelerator device
    config = load_config(config_path)
    set_seed(config.training.seed)
    device = get_device()
    print(f"Using device: {device}")

    # load data module
    data_module = MedMNISTDataModule(
        dataset_name=config.dataset.name,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        download=config.dataset.download,
        as_rgb=config.dataset.as_rgb,
        image_size=config.dataset.image_size,
    )
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    # build model and send to device
    model = build_model(
        identifier=config.model.identifier,
        pretrained=config.model.pretrained,
        source=config.model.source,
        num_classes=data_module.n_classes,
        in_chans=data_module.n_channels,
    ).to(device)

    # initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # initialize model trainer w/ model, optimizer, loss, etc
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=config.artifacts.output_dir,
    )

    # train model
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
    )

    # load the best model before test eval
    best_model_path = fit_result["best_model_path"]
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # evaluate the model and output metrics
    test_metrics = trainer.evaluate(test_loader)
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    # save metrics and confusion matrix
    output_dir = Path(config.artifacts.output_dir)
    save_config(config, output_dir / "config_used.yaml")
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    save_confusion_matrix(
        confusion_matrix=test_metrics["confusion_matrix"],
        output_path=output_dir / "confusion_matrix.png",
    )

    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedMNIST classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    main(args.config)
