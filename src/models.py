import numpy as np
from typing import Optional, Tuple, Dict


class LogisticRegression:
    
    def __init__(self, learning_rate: float = 0.01, 
                 n_iterations: int = 1000,
                 regularization: float = 0.0,
                 verbose: bool = False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = len(y_true)
        
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        if self.weights is not None and self.regularization > 0:
            l2_term = (self.regularization / (2 * m)) * np.sum(self.weights ** 2)
            return bce_loss + l2_term
        
        return bce_loss
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        m, n = X.shape
        
        self.weights = np.zeros(n)
        self.bias = 0
        
        for iteration in range(self.n_iterations):
            # Use einsum for efficient matrix-vector product
            z = np.einsum('ij,j->i', X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            loss = self.compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            error = predictions - y
            
            # Use einsum for gradient computation
            dw = (1/m) * np.einsum('ij,i->j', X, error)
            db = (1/m) * np.sum(error)
            
            if self.regularization > 0:
                dw += (self.regularization / m) * self.weights
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.verbose and (iteration % 100 == 0):
                print(f"Iteration {iteration}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Use einsum for efficient prediction
        z = np.einsum('ij,j->i', X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


class KNearestNeighbors:
    
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
        self.X_train = None
        self.y_train = None
    
    @staticmethod
    def euclidean_distance(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Use einsum for memory-efficient computation
        X1_squared = np.einsum('ij,ij->i', X1, X1)[:, np.newaxis]
        X2_squared = np.einsum('ij,ij->i', X2, X2)[np.newaxis, :]
        
        # Cross term using einsum
        cross_term = np.einsum('ij,kj->ik', X1, X2)
        
        # Numerical stability: ensure non-negative before sqrt
        distances_squared = np.maximum(X1_squared + X2_squared - 2 * cross_term, 0)
        
        return np.sqrt(distances_squared)
    
    @staticmethod
    def manhattan_distance(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_expanded = X1[:, np.newaxis, :]
        X2_expanded = X2[np.newaxis, :, :]
        
        distances = np.sum(np.abs(X1_expanded - X2_expanded), axis=2)
        
        return distances
    
    def compute_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(X1, X2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(X1, X2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self.compute_distances(X, self.X_train)
        
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices].astype(int)
        
        # Vectorized mode calculation using bincount
        # For each row, find the most common label
        n_samples = k_nearest_labels.shape[0]
        n_classes = int(self.y_train.max()) + 1
        
        # Create a 2D array to count votes
        votes = np.zeros((n_samples, n_classes), dtype=int)
        row_indices = np.arange(n_samples)[:, np.newaxis]
        np.add.at(votes, (row_indices, k_nearest_labels), 1)
        
        predictions = np.argmax(votes, axis=1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = self.compute_distances(X, self.X_train)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        probabilities = np.mean(k_nearest_labels, axis=1)
        
        return probabilities


class NaiveBayes:
    
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        m, n = X.shape
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n))
        self.variances = np.zeros((n_classes, n))
        
        for idx, c in enumerate(self.classes):
            mask = (y == c)
            X_c = X[mask]
            
            self.class_priors[idx] = X_c.shape[0] / m
            
            self.means[idx] = np.mean(X_c, axis=0)
            self.variances[idx] = np.var(X_c, axis=0) + self.var_smoothing
        
        return self
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        log_prob = -0.5 * (
            np.log(2 * np.pi * var) + 
            ((x - mean) ** 2) / var
        )
        
        return log_prob
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        n_classes = len(self.classes)
        
        log_probs = np.zeros((m, n_classes))
        
        for idx in range(n_classes):
            log_prior = np.log(self.class_priors[idx])
            
            log_likelihood = np.sum(
                self._gaussian_pdf(X, self.means[idx], self.variances[idx]),
                axis=1
            )
            
            log_probs[:, idx] = log_prior + log_likelihood
        
        return log_probs
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        
        log_probs_shifted = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_shifted)
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        
        class_indices = np.argmax(log_probs, axis=1)
        
        return self.classes[class_indices]


class LinearRegression:
    
    def __init__(self, learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 verbose: bool = False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = len(y_true)
        mse = np.mean((y_pred - y_true) ** 2) / 2
        return mse
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        m, n = X.shape
        
        self.weights = np.zeros(n)
        self.bias = 0
        
        for iteration in range(self.n_iterations):
            # Use einsum for efficient prediction
            predictions = np.einsum('ij,j->i', X, self.weights) + self.bias
            
            loss = self.compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            error = predictions - y
            
            # Use einsum for gradient computation
            dw = (1/m) * np.einsum('ij,i->j', X, error)
            db = (1/m) * np.sum(error)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.verbose and (iteration % 100 == 0):
                print(f"Iteration {iteration}: Loss = {loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Use einsum for efficient prediction
        return np.einsum('ij,j->i', X, self.weights) + self.bias


class ModelEvaluator:
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        n_classes: Optional[int] = None) -> np.ndarray:
        if n_classes is None:
            n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
        
        # Vectorized confusion matrix using bincount
        # Combine true and pred labels into single index
        y_true_int = y_true.astype(int)
        y_pred_int = y_pred.astype(int)
        
        # Use fancy indexing: create linear indices
        indices = y_true_int * n_classes + y_pred_int
        cm_flat = np.bincount(indices, minlength=n_classes * n_classes)
        cm = cm_flat.reshape(n_classes, n_classes)
        
        return cm
    
    @staticmethod
    def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                           average: str = 'binary') -> Dict[str, float]:
        cm = ModelEvaluator.confusion_matrix(y_true, y_pred)
        
        if average == 'binary':
            TP = cm[1, 1]
            FP = cm[0, 1]
            FN = cm[1, 0]
            
            precision = TP / (TP + FP + 1e-10)
            recall = TP / (TP + FN + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            n_classes = cm.shape[0]
            precisions = []
            recalls = []
            f1_scores = []
            
            for i in range(n_classes):
                TP = cm[i, i]
                FP = np.sum(cm[:, i]) - TP
                FN = np.sum(cm[i, :]) - TP
                
                precision = TP / (TP + FP + 1e-10)
                recall = TP / (TP + FN + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            return {
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'f1_score': np.mean(f1_scores)
            }
    
    @staticmethod
    def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        # Vectorized ROC AUC calculation with numerical stability
        sort_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sort_indices]
        
        n_pos = np.sum(y_true == 1) + 1e-10
        n_neg = np.sum(y_true == 0) + 1e-10
        
        # Vectorized cumulative sum
        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg
        
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Trapezoidal integration
        auc = np.trapz(tpr, fpr)
        
        return auc
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return r2


class DecisionTree:
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, 
                 max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
        
    def _gini_impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _split(self, X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float):
        left_mask = X[:, feature_idx] <= threshold
        return left_mask, ~left_mask
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> Tuple:
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in features:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask, right_mask = self._split(X, y, feature_idx, threshold)
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                n = len(y)
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                gini = (n_left/n) * self._gini_impurity(y[left_mask]) + \
                       (n_right/n) * self._gini_impurity(y[right_mask])
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            return {'type': 'leaf', 'value': np.bincount(y.astype(int)).argmax()}
        
        features = np.random.choice(n_features, 
                                   self.max_features if self.max_features else n_features, 
                                   replace=False)
        
        best_feature, best_threshold = self._best_split(X, y, features)
        
        if best_feature is None:
            return {'type': 'leaf', 'value': np.bincount(y.astype(int)).argmax()}
        
        left_mask, right_mask = self._split(X, y, best_feature, best_threshold)
        
        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Dict) -> int:
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Vectorized prediction where possible, fallback to iterative for tree traversal
        predictions = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            predictions[i] = self._predict_sample(x, self.tree)
        return predictions


class RandomForest:
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, max_features: str = 'sqrt',
                 bootstrap: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        
        self.trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y
            
            tree = DecisionTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              max_features=max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Vectorized ensemble prediction
        predictions = np.array([tree.predict(X) for tree in self.trees]).astype(int)
        
        # Transpose to get (n_samples, n_estimators)
        predictions = predictions.T
        
        # Vectorized mode calculation
        n_samples = predictions.shape[0]
        n_classes = int(predictions.max()) + 1
        
        # Create vote matrix
        votes = np.zeros((n_samples, n_classes), dtype=int)
        row_indices = np.arange(n_samples)[:, np.newaxis]
        np.add.at(votes, (row_indices, predictions), 1)
        
        final_predictions = np.argmax(votes, axis=1)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)


def cross_validate(model, X: np.ndarray, y: np.ndarray, 
                   k: int = 5, metric: str = 'accuracy') -> Dict:
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    fold_size = n_samples // k
    scores = []
    
    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else n_samples
        
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        if metric == 'accuracy':
            score = ModelEvaluator.accuracy(y_val, y_pred)
        elif metric == 'f1':
            metrics = ModelEvaluator.precision_recall_f1(y_val, y_pred)
            score = metrics['f1_score']
        else:
            score = ModelEvaluator.accuracy(y_val, y_pred)
        
        scores.append(score)
    
    return {
        'scores': np.array(scores),
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    }

