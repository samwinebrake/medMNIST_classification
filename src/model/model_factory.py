import timm
import torch.nn as nn


def build_model(
    identifier: str,
    pretrained: bool,
    source: str,
    num_classes: int,
    in_chans: int,
    **kwargs,
) -> nn.Module:
    """
    Build and return a classification model.

    Parameters
    ----------
    identifier : str
        Model identifier, for example `convnext_atto.d2_in1k`.
    pretrained : bool
        Whether to load pretrained weights.
    source : str
        Model source library. Currently only ``timm`` is supported.
    num_classes : int
        Number of output classes for the classification head.
    in_chans : int
        Number of input image channels.
    **kwargs
        Additional keyword arguments forwarded to the model library.

    Returns
    -------
    nn.Module
        Instantiated model.
    """
    if source == "timm":
        return timm.create_model(
            identifier,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            **kwargs,
        )

    raise NotImplementedError(f"Unsupported model source: {source}")
