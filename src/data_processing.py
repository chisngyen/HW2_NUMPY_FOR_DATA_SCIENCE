import numpy as np
from typing import Tuple, Dict, List, Optional

class StatisticalAnalyzer:    
    @staticmethod
    def compute_statistics(data: np.ndarray) -> Dict:
        stats = {
            'mean': np.mean(data, axis=0),
            'median': np.median(data, axis=0),
            'std': np.std(data, axis=0),
            'var': np.var(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'range': np.ptp(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q50': np.percentile(data, 50, axis=0),
            'q75': np.percentile(data, 75, axis=0),
        }
        
        mean = stats['mean']
        std = stats['std']
        
        stats['skewness'] = np.mean(((data - mean) / (std + 1e-10)) ** 3, axis=0)
        stats['kurtosis'] = np.mean(((data - mean) / (std + 1e-10)) ** 4, axis=0) - 3
        
        return stats
    
    @staticmethod
    def correlation_matrix(data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        n = data.shape[0]
        # Use einsum for efficient matrix multiplication
        cov_matrix = np.einsum('ij,ik->jk', centered, centered) / (n - 1)
        
        # Numerical stability: avoid division by zero
        std = np.sqrt(np.diag(cov_matrix) + 1e-10)
        corr_matrix = cov_matrix / np.outer(std, std)
        
        return corr_matrix
    
    @staticmethod
    def covariance_matrix(data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=0)
        centered = data - mean
        n = data.shape[0]
        # Use einsum for memory-efficient covariance calculation
        cov_matrix = np.einsum('ij,ik->jk', centered, centered) / (n - 1)
        return cov_matrix

class MissingValueHandler:    
    @staticmethod
    def detect_missing(data: np.ndarray) -> np.ndarray:
        if data.dtype.kind in ['f', 'c']:
            return np.isnan(data)
        
        if data.dtype.kind in ['U', 'S', 'O']:
            mask = (data == '') | (data == 'None') | (data == 'nan')
            return mask
        
        return np.zeros(data.shape, dtype=bool)
    
    @staticmethod
    def fill_mean(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        result = data.copy()
        
        if mask is None:
            mask = np.isnan(data)
        
        col_means = np.nanmean(data, axis=0)
        indices = np.where(mask)
        result[indices] = np.take(col_means, indices[1])
        
        return result
    
    @staticmethod
    def fill_median(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        result = data.copy()
        
        if mask is None:
            mask = np.isnan(data)
        
        col_medians = np.nanmedian(data, axis=0)
        indices = np.where(mask)
        result[indices] = np.take(col_medians, indices[1])
        
        return result
    
    @staticmethod
    def fill_mode(data: np.ndarray) -> np.ndarray:
        result = data.copy()
        
        # Vectorized approach: process all columns at once
        mask = (data == '') | (data == 'None')
        
        # Only process columns with missing values
        cols_with_missing = np.where(np.any(mask, axis=0))[0]
        
        for col_idx in cols_with_missing:
            col_data = data[:, col_idx]
            col_mask = mask[:, col_idx]
            unique_vals, counts = np.unique(col_data[~col_mask], return_counts=True)
            mode_val = unique_vals[np.argmax(counts)]
            result[col_mask, col_idx] = mode_val
        
        return result

class OutlierDetector:    
    @staticmethod
    def z_score_method(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-10))
        outlier_mask = z_scores > threshold
        return outlier_mask
    
    @staticmethod
    def iqr_method(data: np.ndarray, k: float = 1.5) -> np.ndarray:
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        return outlier_mask
    
    @staticmethod
    def remove_outliers(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        row_mask = ~np.any(mask, axis=1)
        return data[row_mask]
    
    @staticmethod
    def cap_outliers(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = data.copy()
        lower = np.percentile(data, 1, axis=0)
        upper = np.percentile(data, 99, axis=0)
        result = np.clip(result, lower, upper)
        return result

class Normalizer:    
    @staticmethod
    def min_max_scaling(data: np.ndarray, 
                       feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Dict]:
        x_min = np.min(data, axis=0)
        x_max = np.max(data, axis=0)
        
        range_diff = x_max - x_min
        range_diff[range_diff == 0] = 1
        scaled = (data - x_min) / range_diff
        
        min_val, max_val = feature_range
        scaled = scaled * (max_val - min_val) + min_val
        
        params = {'min': x_min, 'max': x_max, 'feature_range': feature_range}
        return scaled, params
    
    @staticmethod
    def z_score_standardization(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1
        
        scaled = (data - mean) / std
        params = {'mean': mean, 'std': std}
        return scaled, params
    
    @staticmethod
    def log_transform(data: np.ndarray, offset: float = 1.0) -> np.ndarray:
        return np.log(data + offset)
    
    @staticmethod
    def robust_scaling(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        median = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1
        
        scaled = (data - median) / iqr
        params = {'median': median, 'iqr': iqr}
        return scaled, params

class FeatureEncoder:
    @staticmethod
    def label_encode(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        encoded = np.zeros(data.shape, dtype=np.int32)
        mappings = {}
        
        # Vectorized label encoding per column
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            unique_vals, inverse_indices = np.unique(col_data, return_inverse=True)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            mappings[col_idx] = mapping
            
            # Use inverse_indices directly - fully vectorized
            encoded[:, col_idx] = inverse_indices
        
        return encoded, mappings
    
    @staticmethod
    def one_hot_encode(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        all_encoded = []
        mappings = {}
        
        # Vectorized one-hot encoding
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            unique_vals, inverse_indices = np.unique(col_data, return_inverse=True)
            n_categories = len(unique_vals)
            
            # Vectorized one-hot: use fancy indexing
            one_hot = np.zeros((len(col_data), n_categories), dtype=np.int32)
            one_hot[np.arange(len(col_data)), inverse_indices] = 1
            
            all_encoded.append(one_hot)
            mappings[col_idx] = unique_vals
        
        encoded_data = np.hstack(all_encoded)
        return encoded_data, mappings

class SMOTE:    
    def __init__(self, k_neighbors: int = 5, random_state: Optional[int] = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        minority_count = np.min(counts)
        majority_count = np.max(counts)
        
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        
        n_synthetic = majority_count - minority_count
        synthetic_samples = self._generate_synthetic_samples(X_minority, n_synthetic)
        
        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.concatenate([y, np.full(n_synthetic, minority_class)])
        
        shuffle_idx = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_idx]
        y_resampled = y_resampled[shuffle_idx]
        
        print(f"SMOTE applied: {len(X)} -> {len(X_resampled)} samples")
        print(f"Class distribution: {np.bincount(y_resampled.astype(int))}")
        
        return X_resampled, y_resampled
    
    def _generate_synthetic_samples(self, X_minority: np.ndarray, 
                                   n_synthetic: int) -> np.ndarray:
        n_samples = X_minority.shape[0]
        
        # Vectorized distance calculation with numerical stability
        X_sq = np.sum(X_minority ** 2, axis=1, keepdims=True)
        distances_sq = np.maximum(X_sq + X_sq.T - 2 * np.dot(X_minority, X_minority.T), 0)
        distances = np.sqrt(distances_sq)
        
        # Vectorized sample selection
        sample_indices = np.random.randint(0, n_samples, size=n_synthetic)
        
        # Get k-nearest neighbors for all selected samples
        neighbor_ranks = np.argsort(distances[sample_indices], axis=1)[:, 1:self.k_neighbors+1]
        
        # Randomly select one neighbor from k-nearest for each sample
        neighbor_choices = np.random.randint(0, self.k_neighbors, size=n_synthetic)
        neighbor_indices = neighbor_ranks[np.arange(n_synthetic), neighbor_choices]
        
        # Vectorized synthetic sample generation
        alphas = np.random.random(size=(n_synthetic, 1))
        samples = X_minority[sample_indices]
        neighbors = X_minority[neighbor_indices]
        synthetic_samples = samples + alphas * (neighbors - samples)
        
        return synthetic_samples

class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        n_samples = X.shape[0]
        # Use einsum for efficient covariance matrix
        cov_matrix = np.einsum('ij,ik->jk', X_centered, X_centered) / (n_samples - 1)
        
        # Use eigh for symmetric matrices (more stable than eig)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Numerical stability
        total_var = np.sum(eigenvalues) + 1e-10
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        print(f"PCA fitted with {self.n_components} components")
        print(f"Explained variance ratio: {self.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(self.explained_variance_ratio_):.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        return (X_pca @ self.components_.T) + self.mean_

class FeatureSelector:
    @staticmethod
    def select_by_correlation(X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
        # Fully vectorized correlation calculation
        X_centered = X - np.mean(X, axis=0)
        y_centered = y - np.mean(y)
        
        # Use einsum for efficient computation: sum over samples dimension
        numerator = np.einsum('ij,i->j', X_centered, y_centered)
        
        # Vectorized denominators with numerical stability
        x_norms = np.sqrt(np.sum(X_centered**2, axis=0) + 1e-10)
        y_norm = np.sqrt(np.sum(y_centered**2) + 1e-10)
        
        correlations = np.abs(numerator / (x_norms * y_norm))
        
        selected_mask = correlations >= threshold
        X_selected = X[:, selected_mask]
        selected_names = [name for i, name in enumerate(feature_names) if selected_mask[i]]
        
        print(f"Feature selection by correlation (threshold={threshold}):")
        print(f"  Selected {X_selected.shape[1]} / {X.shape[1]} features")
        
        top_indices = np.argsort(correlations)[-10:][::-1]
        print(f"\n  Top 10 features by correlation:")
        for idx in top_indices[:10]:
            print(f"    {feature_names[idx]}: {correlations[idx]:.4f}")
        
        return X_selected, selected_names
    
    @staticmethod
    def select_by_variance(X: np.ndarray, feature_names: List[str],
                          threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
        variances = np.var(X, axis=0)
        selected_mask = variances >= threshold
        
        X_selected = X[:, selected_mask]
        selected_names = [name for i, name in enumerate(feature_names) if selected_mask[i]]
        
        print(f"Feature selection by variance (threshold={threshold}):")
        print(f"  Selected {X_selected.shape[1]} / {X.shape[1]} features")
        
        return X_selected, selected_names
    
    @staticmethod
    def compute_spearman_correlation(X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        
        # Vectorized ranking using argsort twice
        sorted_indices = np.argsort(X, axis=0)
        X_ranked = np.empty_like(X)
        
        # Vectorized: compute ranks for all columns at once
        X_ranked[sorted_indices, np.arange(n_features)] = np.arange(n_samples)[:, np.newaxis]
        
        # Vectorized correlation matrix calculation using einsum
        mean_ranks = np.mean(X_ranked, axis=0)
        centered_ranks = X_ranked - mean_ranks
        
        # Use einsum for efficient covariance matrix
        cov_matrix = np.einsum('ij,ik->jk', centered_ranks, centered_ranks) / (n_samples - 1)
        
        std = np.sqrt(np.diag(cov_matrix) + 1e-10)
        spearman_corr = cov_matrix / np.outer(std, std)
        
        return spearman_corr

class DataSplitter:
    @staticmethod
    def train_test_split(X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> Tuple:
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def k_fold_split(X: np.ndarray, k: int = 5, 
                    random_state: Optional[int] = None) -> List[Tuple]:
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        
        current = 0
        folds = []
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            folds.append((train_indices, val_indices))
            current = stop
        
        return folds


def save_processed_data(data: np.ndarray, filepath: str):
    np.save(filepath, data)
    print(f"Saved processed data to {filepath}")


def load_processed_data(filepath: str) -> np.ndarray:
    data = np.load(filepath)
    print(f"Loaded processed data from {filepath}")
    return data