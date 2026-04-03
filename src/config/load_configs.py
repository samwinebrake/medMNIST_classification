from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    """
    Configuration settings for dataset loading and preprocessing.

    Attributes
    ----------
    name : str
        Name of the MedMNIST dataset to load.
    image_size : int
        Target image size used for resizing input images.
    batch_size : int
        Batch size used by the dataloaders.
    download : bool
        Whether the dataset should be downloaded if it is not already present.
    as_rgb : bool
        Whether images should be converted to RGB format.
    num_workers : int, default=4
        Number of worker processes used by the dataloaders.
    """

    name: str
    image_size: int
    batch_size: int
    download: bool
    as_rgb: bool
    num_workers: int = 4


@dataclass
class ModelConfig:
    """
    Configuration settings for model construction.

    Attributes
    ----------
    identifier : str
        Model identifier passed to the model factory.
    pretrained : bool
        Whether pretrained weights should be loaded.
    source : str
        Source library from which the model should be created.
        Currently only ``timm`` is supported.
    """

    identifier: str
    pretrained: bool
    source: str


@dataclass
class TrainingConfig:
    """
    Configuration settings for model training.

    Attributes
    ----------
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate used by the optimizer.
    weight_decay : float, default=1e-4
        Weight decay used by the optimizer.
    seed : int, default=42
        Random seed used for reproducibility.
    """

    num_epochs: int
    lr: float
    weight_decay: float = 1e-4
    seed: int = 42


@dataclass
class ArtifactsConfig:
    """
    Configuration settings for output artifacts.

    Attributes
    ----------
    output_dir : str, default="artifacts/run"
        Directory where training artifacts should be saved.
    """

    output_dir: str = "artifacts/run"


@dataclass
class AppConfig:
    """
    Top-level application configuration.

    Attributes
    ----------
    dataset : DatasetConfig
        Dataset-related configuration.
    model : ModelConfig
        Model-related configuration.
    training : TrainingConfig
        Training-related configuration.
    artifacts : ArtifactsConfig
        Artifact output-related configuration.
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig


def load_config(config_path: str | Path) -> AppConfig:
    """
    Load a YAML configuration file into a structured application config.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    AppConfig
        Parsed application configuration as nested dataclass objects.
    """
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        artifacts=ArtifactsConfig(**raw.get("artifacts", {})),
    )


def save_config(config: AppConfig, output_path: str | Path):
    """
    Save an application configuration object to a YAML file.

    Parameters
    ----------
    config : AppConfig
        Configuration object to serialize and save.
    output_path : str or pathlib.Path
        Destination path for the YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
