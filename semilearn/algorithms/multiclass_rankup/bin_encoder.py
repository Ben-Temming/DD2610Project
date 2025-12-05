import numpy as np
import torch


class BinEncoder:
    """
    Encodes continuous values (e.g., ages) into discrete bin classes.
    
    Supports two strategies:
    - "equal_width": Bins have equal range (e.g., 1-15, 16-30, ...)
    - "quantile": Bins have equal number of samples (computed from labeled training data)
    
    Args:
        num_classes (int): Number of bins/classes
        strategy (str): Binning strategy - "equal_width" or "quantile"
    """
    
    def __init__(self, num_classes, strategy="equal_width"):
        if strategy not in ["equal_width", "quantile"]:
            raise ValueError(f"Unknown strategy: {strategy}. Must be 'equal_width' or 'quantile'")
        
        self.num_classes = num_classes
        self.strategy = strategy
        self.bin_edges = None
        self.fitted = False
    

    def fit(self, labels):
        """
        Compute bin edges from labeled data.
        
        Args:
            labels: numpy array or tensor of continuous values (e.g., ages)
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels).flatten()
        
        if self.strategy == "quantile":
            raise NotImplementedError("Quantile binning is currently disabled. Because it does not work well with the small amount of labeled data in semi-supervised learning.")
            # Equal-frequency bins: each bin has approximately the same number of samples
            quantiles = np.linspace(0, 100, self.num_classes + 1)
            self.bin_edges = np.percentile(labels, quantiles)
            # Handle duplicate edges (can happen with discrete or concentrated data)
            # Make edges strictly increasing by adding small epsilon
            eps = 1e-6
            for i in range(1, len(self.bin_edges)):
                if self.bin_edges[i] <= self.bin_edges[i-1]:
                    self.bin_edges[i] = self.bin_edges[i-1] + eps
            # Ensure edges span the full range
            self.bin_edges = np.unique(self.bin_edges)
            # If we lost bins due to duplicates, redistribute
            if len(self.bin_edges) < self.num_classes + 1:
                # Fall back to equal-width for the remaining range
                self.bin_edges = np.linspace(labels.min(), labels.max(), self.num_classes + 1)
                    
        elif self.strategy == "equal_width":
            # Equal-width bins: each bin covers the same range
            min_val, max_val = labels.min(), labels.max()
            self.bin_edges = np.linspace(min_val, max_val, self.num_classes + 1)
        
        # Ensure first and last edges capture all possible values
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        
        self.fitted = True
        return self
    

    def transform(self, values):
        """
        Convert continuous values to bin class indices.
        
        Args:
            values: tensor or numpy array of continuous values
            
        Returns:
            Tensor of class indices (0 to num_classes-1)
        """
        if not self.fitted:
            raise RuntimeError("BinEncoder must be fitted before transform. Call fit() first.")
        
        is_tensor = isinstance(values, torch.Tensor)
        device = values.device if is_tensor else None
        
        if is_tensor:
            values_np = values.cpu().numpy()
        else:
            values_np = np.asarray(values)
        
        # np.digitize returns bin index (1-indexed), subtract 1 for 0-indexed classes
        # Also clip to valid range [0, num_classes-1]
        bin_indices = np.digitize(values_np, self.bin_edges[1:-1])  # Use inner edges
        bin_indices = np.clip(bin_indices, 0, self.num_classes - 1)
        
        if is_tensor:
            return torch.tensor(bin_indices, dtype=torch.long, device=device)
        return bin_indices
    

    def get_bin_edges(self):
        """Return the computed bin edges (useful for logging/debugging)."""
        if not self.fitted:
            raise RuntimeError("BinEncoder must be fitted before getting bin edges. Call fit() first.")
        
        return self.bin_edges.copy()
    

    def get_bin_info(self, scaler=None, original_range=None):
        """
        Return a formatted string describing the bins.
        
        Args:
            scaler: Optional scaler to convert bin edges back to original scale
            original_range: Optional tuple (min, max) of original range for display
        """
        if not self.fitted:
            raise RuntimeError("BinEncoder must be fitted before getting bin info. Call fit() first.")
        
        info_lines = [f"BinEncoder (strategy={self.strategy}, num_classes={self.num_classes})"]
        if original_range is not None:
            info_lines.append(f"Note: Values are scaled to [0, 1]. Original range: {original_range}")
        info_lines.append("-" * 50)
        
        for i in range(self.num_classes):
            left = self.bin_edges[i]
            right = self.bin_edges[i + 1]
            left_str = f"{left:.3f}" if left != -np.inf else "-inf"
            right_str = f"{right:.3f}" if right != np.inf else "inf"
            
            # If scaler provided, show original values too
            if scaler is not None:
                left_orig = scaler.inverse_transform(torch.tensor([left])).item() if left != -np.inf else "-inf"
                right_orig = scaler.inverse_transform(torch.tensor([right])).item() if right != np.inf else "inf"
                if isinstance(left_orig, float):
                    left_orig = f"{left_orig:.1f}"
                if isinstance(right_orig, float):
                    right_orig = f"{right_orig:.1f}"
                info_lines.append(f"  Class {i}: scaled ({left_str}, {right_str}] = original ({left_orig}, {right_orig}]")
            else:
                info_lines.append(f"  Class {i}: ({left_str}, {right_str}]")
        
        return "\n".join(info_lines)
    

    def __repr__(self):
        """String representation of the BinEncoder."""
        status = "fitted" if self.fitted else "not fitted"
        return f"BinEncoder(num_classes={self.num_classes}, strategy='{self.strategy}', {status})"
