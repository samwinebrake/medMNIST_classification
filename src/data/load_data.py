import medmnist
import torch
import torchvision.transforms as transforms
from medmnist import INFO
from torch.utils.data import DataLoader


class MedMNISTDataModule:
    """
    Build MedMNIST datasets, transforms, and dataloaders for training and evaluation.

    Parameters
    ----------
    dataset_name : str
        Name of the MedMNIST dataset to load.
    batch_size : int, default=64
        Batch size used for all dataloaders.
    num_workers : int, default=4
        Number of worker processes used by the dataloaders.
    download : bool, default=True
        Whether the dataset should be downloaded if it is not already present.
    as_rgb : bool, default=False
        Whether images should be converted to RGB format.
    image_size : int, default=28
        Target image size used for resizing.

    Attributes
    ----------
    dataset_name : str
        Name of the MedMNIST dataset.
    batch_size : int
        Batch size used for loading data.
    num_workers : int
        Number of worker processes used by dataloaders.
    download : bool
        Whether dataset downloading is enabled.
    as_rgb : bool
        Whether images are converted to RGB.
    image_size : int
        Target size used for image resizing.
    info : dict
        Metadata dictionary retrieved from `medmnist.INFO` for the selected dataset.
    n_channels : int
        Number of model input channels after optional RGB conversion.
    n_classes : int
        Number of classification output classes.
    data_class : type
        MedMNIST dataset class corresponding to `dataset_name`.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 64,
        num_workers: int = 4,
        download: bool = True,
        as_rgb: bool = False,
        image_size: int = 28,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.as_rgb = as_rgb
        self.image_size = image_size

        self.info = INFO[dataset_name]
        self.n_channels = 3 if self.as_rgb else self.info["n_channels"]
        self.n_classes = len(self.info["label"])
        self.data_class = getattr(medmnist, self.info["python_class"])

    def _get_normalization(self):
        """
        Return channel-wise normalization statistics.

        For RGB inputs, ImageNet normalization is used to better align with
        ImageNet-pretrained backbones. For grayscale inputs, a simple
        mean/std pair of 0.5 is used.

        Returns
        -------
        tuple[tuple[float, ...], tuple[float, ...]]
            Mean and standard deviation tuples used for normalization.
        """
        if self.as_rgb:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return (0.5,), (0.5,)

    def create_transforms(self):
        """
        Create training and evaluation transforms.

        The training transform includes lightweight augmentation, while the
        evaluation transform is deterministic. Both apply resizing, tensor
        conversion, and normalization.

        Returns
        -------
        tuple[torchvision.transforms.Compose, torchvision.transforms.Compose]
            Training transform and evaluation transform.
        """
        normalize_mean, normalize_std = self._get_normalization()

        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalize_mean,
                    std=normalize_std,
                ),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalize_mean,
                    std=normalize_std,
                ),
            ]
        )

        return train_transform, eval_transform

    def get_datasets(self):
        """
        Build train, validation, and test dataset objects.

        Returns
        -------
        tuple
            Train, validation, and test MedMNIST dataset objects.
        """
        train_transform, eval_transform = self.create_transforms()

        train_dataset = self.data_class(
            split="train",
            transform=train_transform,
            download=self.download,
            as_rgb=self.as_rgb,
            size=self.image_size,
        )
        val_dataset = self.data_class(
            split="val",
            transform=eval_transform,
            download=self.download,
            as_rgb=self.as_rgb,
            size=self.image_size,
        )
        test_dataset = self.data_class(
            split="test",
            transform=eval_transform,
            download=self.download,
            as_rgb=self.as_rgb,
            size=self.image_size,
        )

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(self):
        """
        Build train, validation, and test dataloaders.

        Returns
        -------
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
            Train, validation, and test dataloaders.
        """
        train_dataset, val_dataset, test_dataset = self.get_datasets()
        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader
