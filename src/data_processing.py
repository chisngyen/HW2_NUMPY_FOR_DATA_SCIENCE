import numpy as np
from typing import Tuple, Dict, List, Optional

class StatisticalAnalyzer:    
    @staticmethod
    def compute_statistics(data: np.ndarray) -> Dict:
        """Compute descriptive statistics for dataset"""
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
        """Compute correlation matrix"""
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        n = data.shape[0]
        cov_matrix = (centered.T @ centered) / (n - 1)
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std, std)
        
        return corr_matrix
    
    @staticmethod
    def covariance_matrix(data: np.ndarray) -> np.ndarray:
        """Compute covariance matrix"""
        mean = np.mean(data, axis=0)
        centered = data - mean
        n = data.shape[0]
        cov_matrix = (centered.T @ centered) / (n - 1)
        return cov_matrix

class MissingValueHandler:    
    @staticmethod
    def detect_missing(data: np.ndarray) -> np.ndarray:
        """Detect missing values (NaN, None, empty strings)"""
        if data.dtype.kind in ['f', 'c']:
            return np.isnan(data)
        
        if data.dtype.kind in ['U', 'S', 'O']:
            mask = (data == '') | (data == 'None') | (data == 'nan')
            return mask
        
        return np.zeros(data.shape, dtype=bool)
    
    @staticmethod
    def fill_mean(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Fill missing values with column mean"""
        result = data.copy()
        
        if mask is None:
            mask = np.isnan(data)
        
        col_means = np.nanmean(data, axis=0)
        indices = np.where(mask)
        result[indices] = np.take(col_means, indices[1])
        
        return result
    
    @staticmethod
    def fill_median(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Fill missing values with column median"""
        result = data.copy()
        
        if mask is None:
            mask = np.isnan(data)
        
        col_medians = np.nanmedian(data, axis=0)
        indices = np.where(mask)
        result[indices] = np.take(col_medians, indices[1])
        
        return result
    
    @staticmethod
    def fill_mode(data: np.ndarray) -> np.ndarray:
        """Fill missing values with mode for categorical data"""
        result = data.copy()
        
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            mask = (col_data == '') | (col_data == 'None')
            
            if np.any(mask):
                unique_vals, counts = np.unique(col_data[~mask], return_counts=True)
                mode_val = unique_vals[np.argmax(counts)]
                result[mask, col_idx] = mode_val
        
        return result

class OutlierDetector:    
    @staticmethod
    def z_score_method(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-10))
        outlier_mask = z_scores > threshold
        return outlier_mask
    
    @staticmethod
    def iqr_method(data: np.ndarray, k: float = 1.5) -> np.ndarray:
        """Detect outliers using IQR method"""
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        return outlier_mask
    
    @staticmethod
    def remove_outliers(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove rows containing outliers"""
        row_mask = ~np.any(mask, axis=1)
        return data[row_mask]
    
    @staticmethod
    def cap_outliers(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Cap outliers at percentile bounds"""
        result = data.copy()
        lower = np.percentile(data, 1, axis=0)
        upper = np.percentile(data, 99, axis=0)
        result = np.clip(result, lower, upper)
        return result

class Normalizer:    
    @staticmethod
    def min_max_scaling(data: np.ndarray, 
                       feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Dict]:
        """Min-Max normalization to specified range"""
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
        """Z-score standardization (mean=0, std=1)"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1
        
        scaled = (data - mean) / std
        params = {'mean': mean, 'std': std}
        return scaled, params
    
    @staticmethod
    def log_transform(data: np.ndarray, offset: float = 1.0) -> np.ndarray:
        """Log transformation for skewed distributions"""
        return np.log(data + offset)
    
    @staticmethod
    def robust_scaling(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Robust scaling using median and IQR (resistant to outliers)"""
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
        """Label encoding for categorical variables"""
        encoded = np.zeros(data.shape, dtype=np.int32)
        mappings = {}
        
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            unique_vals = np.unique(col_data)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            mappings[col_idx] = mapping
            
            for val, idx in mapping.items():
                mask = col_data == val
                encoded[mask, col_idx] = idx
        
        return encoded, mappings
    
    @staticmethod
    def one_hot_encode(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """One-hot encoding for categorical variables"""
        all_encoded = []
        mappings = {}
        
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            unique_vals = np.unique(col_data)
            n_categories = len(unique_vals)
            
            one_hot = np.zeros((len(col_data), n_categories), dtype=np.int32)
            
            for idx, val in enumerate(unique_vals):
                mask = col_data == val
                one_hot[mask, idx] = 1
            
            all_encoded.append(one_hot)
            mappings[col_idx] = unique_vals
        
        encoded_data = np.hstack(all_encoded)
        return encoded_data, mappings

class SMOTE:    
    def __init__(self, k_neighbors: int = 5, random_state: Optional[int] = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample minority class using SMOTE algorithm"""
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
        """Generate synthetic samples using k-NN interpolation"""
        n_samples = X_minority.shape[0]
        synthetic_samples = []
        
        X_sq = np.sum(X_minority ** 2, axis=1, keepdims=True)
        distances = np.sqrt(X_sq + X_sq.T - 2 * np.dot(X_minority, X_minority.T))
        
        for _ in range(n_synthetic):
            idx = np.random.randint(0, n_samples)
            sample = X_minority[idx]
            
            neighbor_indices = np.argsort(distances[idx])[1:self.k_neighbors+1]
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_minority[neighbor_idx]
            
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        return np.array(synthetic_samples)

class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """Fit PCA model using eigendecomposition"""
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        print(f"PCA fitted with {self.n_components} components")
        print(f"Explained variance ratio: {self.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(self.explained_variance_ratio_):.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to PCA space"""
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """Transform back from PCA space to original space"""
        return (X_pca @ self.components_.T) + self.mean_

class FeatureSelector:
    @staticmethod
    def select_by_correlation(X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
        """Select features based on correlation with target"""
        correlations = []
        
        for i in range(X.shape[1]):
            x_centered = X[:, i] - np.mean(X[:, i])
            y_centered = y - np.mean(y)
            
            corr = np.sum(x_centered * y_centered) / (
                np.sqrt(np.sum(x_centered**2)) * np.sqrt(np.sum(y_centered**2))
            )
            correlations.append(abs(corr))
        
        correlations = np.array(correlations)
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
        """Remove low-variance features"""
        variances = np.var(X, axis=0)
        selected_mask = variances >= threshold
        
        X_selected = X[:, selected_mask]
        selected_names = [name for i, name in enumerate(feature_names) if selected_mask[i]]
        
        print(f"Feature selection by variance (threshold={threshold}):")
        print(f"  Selected {X_selected.shape[1]} / {X.shape[1]} features")
        
        return X_selected, selected_names
    
    @staticmethod
    def compute_spearman_correlation(X: np.ndarray) -> np.ndarray:
        """Compute Spearman rank correlation matrix"""
        n_samples, n_features = X.shape
        X_ranked = np.zeros_like(X)
        
        for i in range(n_features):
            sorted_indices = np.argsort(X[:, i])
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(n_samples)
            X_ranked[:, i] = ranks
        
        mean_ranks = np.mean(X_ranked, axis=0)
        centered_ranks = X_ranked - mean_ranks
        
        cov_matrix = (centered_ranks.T @ centered_ranks) / (n_samples - 1)
        std = np.sqrt(np.diag(cov_matrix))
        spearman_corr = cov_matrix / np.outer(std, std)
        
        return spearman_corr

class DataSplitter:
    @staticmethod
    def train_test_split(X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> Tuple:
        """Split data into training and testing sets"""
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
        """K-Fold cross-validation split"""
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