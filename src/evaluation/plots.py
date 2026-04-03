from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(confusion_matrix, output_path: str | Path, class_labels=None):
    """
    Save a confusion matrix visualization to disk.

    Parameters
    ----------
    confusion_matrix : array-like
        Confusion matrix values. This may be a NumPy array or a nested list,
        such as one loaded from JSON.
    output_path : str or pathlib.Path
        Destination path for the saved image.
    class_labels : array-like, optional
        Labels used for axis tick marks. If not provided, integer indices
        are used.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confusion_matrix = np.array(confusion_matrix)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_labels,
    )
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
