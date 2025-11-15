import numpy as np
from typing import Tuple, Dict, List, Optional

class StatisticalAnalyzer:    
    @staticmethod
    def compute_statistics(data: np.ndarray) -> Dict:
        """
        Compute descriptive statistics (vectorized)
        
        Returns dictionary with:
            - mean, median, mode
            - variance, std
            - min, max, range
            - quantiles (25%, 50%, 75%)
            - skewness, kurtosis
        """
        stats = {
            'mean': np.mean(data, axis=0),
            'median': np.median(data, axis=0),
            'std': np.std(data, axis=0),
            'var': np.var(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'range': np.ptp(data, axis=0),  # peak-to-peak
            'q25': np.percentile(data, 25, axis=0),
            'q50': np.percentile(data, 50, axis=0),
            'q75': np.percentile(data, 75, axis=0),
        }
        
        # Skewness (vectorized)
        mean = stats['mean']
        std = stats['std']
        n = data.shape[0]
        
        # Formula: skewness = E[((X - μ) / σ)³]
        stats['skewness'] = np.mean(((data - mean) / (std + 1e-10)) ** 3, axis=0)
        
        # Kurtosis (vectorized)
        # Formula: kurtosis = E[((X - μ) / σ)⁴] - 3
        stats['kurtosis'] = np.mean(((data - mean) / (std + 1e-10)) ** 4, axis=0) - 3
        
        return stats
    
    @staticmethod
    def correlation_matrix(data: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix using NumPy
        
        Formula: corr(X,Y) = cov(X,Y) / (σ_X * σ_Y)
        
        Pure NumPy implementation without np.corrcoef
        """
        # Center the data
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        # Compute covariance matrix
        n = data.shape[0]
        cov_matrix = (centered.T @ centered) / (n - 1)
        
        # Compute standard deviations
        std = np.sqrt(np.diag(cov_matrix))
        
        # Compute correlation matrix
        # Broadcasting: divide by outer product of stds
        corr_matrix = cov_matrix / np.outer(std, std)
        
        return corr_matrix
    
    @staticmethod
    def covariance_matrix(data: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix using NumPy
        
        Formula: cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]
        """
        # Center the data
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        # Compute covariance using matrix multiplication
        n = data.shape[0]
        cov_matrix = (centered.T @ centered) / (n - 1)
        
        return cov_matrix