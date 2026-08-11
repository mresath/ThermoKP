"""
===========================================================================
ThermoKP Tensor Dataset
Description: PyTorch Geometric Dataset for ThermoKP Tensors
===========================================================================

Workflow:
1. Locates all `.pt` PyG tensor files in the specified root directory.
2. Loads them lazily into memory when queried.

Known Caveats:
- Data is loaded lazily which might slow down fast dataloading, but ensures memory limits aren't exceeded.

Author: ThermoKP Team
License: MIT
"""

from pathlib import Path
import torch
from torch_geometric.data import Dataset

# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Class
# ═══════════════════════════════════════════════════════════════════════════

class EnzymeDataset(Dataset):
    """
    PyTorch Geometric Dataset to load saved HeteroData tensors.

    Attributes
    ----------
    root_dir : pathlib.Path
        Path to the directory containing `.pt` files.
    files : list of pathlib.Path
        List of paths to the `.pt` files.
    """
    def __init__(self, root_dir: str) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        root_dir : str
            Path to the directory containing `.pt` files.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.pt"))

    def len(self) -> int:
        """
        Get the number of items in the dataset.

        Returns
        -------
        int
            Number of items.
        """
        return len(self.files)

    def get(self, idx: int):
        """
        Get the tensor at index idx.

        Parameters
        ----------
        idx : int
            The index of the item.

        Returns
        -------
        torch_geometric.data.HeteroData
            The loaded tensor.
        """
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        if "protein_sequence" in data.node_types and hasattr(data["protein_sequence"], "aa_indices"):
            data["protein_sequence"].num_nodes = data["protein_sequence"].aa_indices.size(0)
        return data
