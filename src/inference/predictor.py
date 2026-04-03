from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config.load_configs import load_config
from src.data.load_data import MedMNISTDataModule
from src.model.model_factory import build_model


class Predictor:
    """
    Load a trained model and run inference on individual images.

    Parameters
    ----------
    config_path : str, default="configs/config.yaml"
        Path to the YAML configuration file used to define dataset,
        model, and artifact settings.
    checkpoint_path : str or None, default=None
        Path to a model checkpoint. If not provided, the predictor
        loads ``best_model.pt`` from the configured artifact directory.
    device : str or None, default=None
        Device override for inference. If not provided, the best
        available device is selected automatically.

    Attributes
    ----------
    config : AppConfig
        Loaded application configuration.
    device : torch.device
        Device used for inference.
    data_module : MedMNISTDataModule
        Data module used to recover dataset metadata and evaluation transforms.
    eval_transform : torchvision.transforms.Compose
        Deterministic preprocessing pipeline used during inference.
    model : torch.nn.Module
        Classification model loaded with trained weights.
    checkpoint_path : pathlib.Path
        Path to the checkpoint currently loaded into the model.
    class_labels : dict or list
        Class-label mapping taken from MedMNIST metadata.
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        checkpoint_path: str | None = None,
        device: str | None = None,
    ):
        self.config = load_config(config_path)
        self.device = self._get_device(device)

        # build data module so we can reuse dataset metadata + eval transforms
        self.data_module = MedMNISTDataModule(
            dataset_name=self.config.dataset.name,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            download=self.config.dataset.download,
            as_rgb=self.config.dataset.as_rgb,
            image_size=self.config.dataset.image_size,
        )

        _, self.eval_transform = self.data_module.create_transforms()

        # build the model that will be injected w/ weights
        self.model = build_model(
            identifier=self.config.model.identifier,
            pretrained=self.config.model.pretrained,
            source=self.config.model.source,
            num_classes=self.data_module.n_classes,
            in_chans=self.data_module.n_channels,
        ).to(self.device)

        # pull in the checkpoint from training
        if checkpoint_path is None:
            checkpoint_path = Path(self.config.artifacts.output_dir) / "best_model.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # load weights into model
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        self.class_labels = self.data_module.info["label"]

    def _get_device(self, device: str | None) -> torch.device:
        """
        Return the device used for inference.

        Parameters
        ----------
        device : str or None
            Optional device override. If provided, it is used directly.
            Otherwise, the best available device is selected automatically.

        Returns
        -------
        torch.device
            Device used for model inference.
        """
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single image for model inference.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image to preprocess.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor with a batch dimension added and
            moved to the configured inference device.
        """
        if image.mode != "RGB" and self.config.dataset.as_rgb:
            image = image.convert("RGB")

        tensor = self.eval_transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """
        Run inference on a single image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image to classify.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the predicted class index, class label,
            confidence score, class probabilities, and checkpoint path.
        """
        input_tensor = self.preprocess_image(image)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

        return {
            "predicted_class": predicted_class,
            "predicted_label": (
                self.class_labels[str(predicted_class)]
                if isinstance(self.class_labels, dict)
                else self.class_labels[predicted_class]
            ),
            "confidence": confidence,
            "probabilities": probabilities.cpu().tolist(),
            "checkpoint_path": str(self.checkpoint_path),
        }
