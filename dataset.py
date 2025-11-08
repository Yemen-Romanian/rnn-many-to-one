import numpy as np
from torch.utils.data import Dataset

def generate_synthetic_sequence_regression_data(num_samples, 
                                                sequence_length, 
                                                num_features, 
                                                noise_level=0.1,
                                                seed=0):
    """
    Generate synthetic data for sequence regression tasks.

    Parameters:
    - num_samples: int, number of samples to generate
    - sequence_length: int, length of each sequence
    - num_features: int, number of features in each time step
    - noise_level: float, standard deviation of Gaussian noise added to the target
    - seed: int, random seed for reproducibility

    Returns:
    - X: np.ndarray of shape (num_samples, sequence_length, num_features), input sequences
    - y: np.ndarray of shape (num_samples,), target values
    """
    # Generate random input sequences
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=-1.0, high=1.0, size=(num_samples, sequence_length, num_features))
    a = X[:, :, 0]  # First feature across all time steps
    b = X[:, :, 1]  # Second feature across all time steps
    c = X[:, :, 2]  # Third feature across all time steps

    y = np.sin(a.mean(axis=1)) + np.abs(b.sum(axis=1)) + np.sqrt(np.abs(c.max(axis=1)))
    y += rng.normal(loc=0.0, scale=noise_level, size=num_samples)
    return X, y


class SyntheticSequenceRegressionDataset(Dataset):
    """
    PyTorch Dataset for synthetic sequence regression data.
    """
    def __init__(self, num_samples, sequence_length, num_features, noise_level=0.1, seed=0):
        self.X, self.y = generate_synthetic_sequence_regression_data(
            num_samples, sequence_length, num_features, noise_level, seed
        )

        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

