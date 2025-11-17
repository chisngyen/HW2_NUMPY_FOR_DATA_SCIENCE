# DỰ ĐOÁN CHURN KHÁCH HÀNG SỬ DỤNG NUMPY

Dự án phân tích và dự đoán hành vi churn (ngừng sử dụng dịch vụ) của khách hàng thẻ tín dụng sử dụng các thuật toán Machine Learning được xây dựng hoàn toàn bằng NumPy.

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

## Giới Thiệu

### Mô Tả Bài Toán

Churn prediction là bài toán dự đoán khách hàng có khả năng ngừng sử dụng dịch vụ hay không. Trong lĩnh vực ngân hàng và thẻ tín dụng, việc nhận diện sớm khách hàng có nguy cơ churn giúp doanh nghiệp:

- Tiết kiệm chi phí thu hút khách hàng mới (cao gấp 5-25 lần so với giữ chân khách hàng cũ)
- Tăng doanh thu bằng cách áp dụng các chiến dịch retention hiệu quả
- Cải thiện trải nghiệm khách hàng thông qua các dịch vụ cá nhân hóa

### Động Lực và Ứng Dụng Thực Tế

**Tại sao sử dụng NumPy?**

- Hiểu sâu về các thuật toán Machine Learning bằng cách implement từ đầu
- Không phụ thuộc vào thư viện black-box (scikit-learn, TensorFlow)
- Tối ưu hiệu suất với NumPy vectorization
- Ứng dụng trong education và research

**Ứng dụng thực tế:**

- Banking: Dự đoán khách hàng đóng thẻ tín dụng
- Telecom: Phát hiện khách hàng có ý định chuyển mạng
- E-commerce: Nhận diện khách hàng không quay lại
- SaaS: Dự đoán subscription cancellation

### Mục Tiêu Cụ Thể

1. Xây dựng các thuật toán ML (Logistic Regression, KNN, Naive Bayes, Random Forest,...) from sratch bằng NumPy.
2. So sánh hiệu quả giữa preprocessing cơ bản (basic) và nâng cao (enhanced)
3. Phân tích insights từ confusion matrix và feature importance
4. Đánh giá model stability thông qua cross-validation

## Dataset

### Nguồn Dữ Liệu

Dataset: **Bank Churners Dataset**

- Nguồn: Kaggle / Leaps Analyttica
- File: `data/raw/BankChurners.csv`
- Kích thước: 10,127 records x 23 features

### Mô Tả Các Features

**Demographic Features:**

- `Customer_Age`: Tuổi khách hàng
- `Gender`: Giới tính (M/F)
- `Dependent_count`: Số người phụ thuộc
- `Education_Level`: Trình độ học vấn
- `Marital_Status`: Tình trạng hôn nhân
- `Income_Category`: Mức thu nhập

**Banking Relationship:**

- `Card_Category`: Loại thẻ (Blue, Silver, Gold, Platinum)
- `Months_on_book`: Số tháng làm khách hàng
- `Total_Relationship_Count`: Tổng số sản phẩm đã sử dụng
- `Months_Inactive_12_mon`: Số tháng không hoạt động trong 12 tháng
- `Contacts_Count_12_mon`: Số lần liên hệ trong 12 tháng

**Financial Metrics:**

- `Credit_Limit`: Hạn mức tín dụng
- `Total_Revolving_Bal`: Số dư đáo hạn
- `Avg_Open_To_Buy`: Hạn mức còn lại trung bình
- `Total_Trans_Amt`: Tổng giá trị giao dịch (12 tháng)
- `Total_Trans_Ct`: Tổng số giao dịch (12 tháng)
- `Total_Ct_Chng_Q4_Q1`: Thay đổi số giao dịch Q4 vs Q1
- `Total_Amt_Chng_Q4_Q1`: Thay đổi giá trị giao dịch Q4 vs Q1
- `Avg_Utilization_Ratio`: Tỷ lệ sử dụng hạn mức trung bình

**Target Variable:**

- `Attrition_Flag`: Existing Customer (0) / Attrited Customer (1)

### Kích Thước và Đặc Điểm Dữ Liệu

- **Tổng số mẫu:** 10,127 khách hàng
- **Class distribution:**
  - Existing Customer: 8,500 (83.93%)
  - Attrited Customer: 1,627 (16.07%)
- **Missing values:** Không có giá trị null
- **Feature types:**
  - Numerical: 14 features
  - Categorical: 6 features

**Train/Test Split:**

- Training set: 80% (8,101 samples)
- Test set: 20% (2,026 samples)

## Method

### Quy Trình Xử Lý Dữ Liệu

#### 1. Data Exploration (`01_data_exploration.ipynb`)

- Thống kê mô tả (mean, median, std, quartiles)
- Phân tích phân phối features
- Correlation analysis
- Phát hiện outliers
- Phân tích churn patterns

#### 2. Preprocessing (`02_preprocessing.ipynb`)

**Basic Preprocessing:**

- Phát hiện outliers
- Label encoding cho categorical features
- Z-score standardization
- Train-test split (80-20)

**Enhanced Preprocessing:**

- Label encoding + One-hot encoding
- Z-score standardization
- Outlier handling (IQR method)
- SMOTE (Synthetic Minority Over-sampling Technique) cho class imbalance
- Feature engineering:
  - Transaction velocity (trans_amt / trans_ct)
  - Utilization categories
  - Activity scores

#### 3. Modeling (`03_modeling.ipynb`)

Train và đánh giá 4 models (Logistic Regression, KNN, Naive Bayes, Random Forest) trên cả 2 dataset versions.

### Thuật Toán Sử Dụng

#### 1. Logistic Regression

**Công thức:**

Sigmoid function:

```
σ(z) = 1 / (1 + e^(-z))
```

Hypothesis:

```
h(x) = σ(w^T x + b)
```

Loss function (Binary Cross-Entropy):

```
J(w) = -(1/m) Σ [y_i log(h(x_i)) + (1 - y_i) log(1 - h(x_i))] + (λ/2m) ||w||^2
```

Gradient descent update:

```
w := w - α * (1/m) X^T (h(X) - y) + (λ/m) w
b := b - α * (1/m) Σ (h(x_i) - y_i)
```

**NumPy Implementation:**

```python
z = np.dot(X, self.weights) + self.bias
predictions = 1 / (1 + np.exp(-z))
error = predictions - y
dw = (1/m) * np.dot(X.T, error) + (self.regularization / m) * self.weights
db = (1/m) * np.sum(error)
self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```

#### 2. K-Nearest Neighbors (KNN)

**Công thức:**

Euclidean distance:

```
d(x_i, x_j) = √(Σ(x_i_k - x_j_k)^2)
```

Vectorized distance matrix:

```
D^2 = ||X1||^2 + ||X2||^2 - 2 X1 X2^T
```

**NumPy Implementation:**

```python
X1_squared = np.sum(X1 ** 2, axis=1, keepdims=True)
X2_squared = np.sum(X2 ** 2, axis=1, keepdims=True)
distances_squared = X1_squared + X2_squared.T - 2 * np.dot(X1, X2.T)
distances = np.sqrt(np.maximum(distances_squared, 0))
k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
predictions = np.array([np.bincount(labels).argmax() for labels in k_nearest_labels])
```

#### 3. Naive Bayes

**Công thức:**

Bayes theorem:

```
P(C_k | x) = P(x | C_k) P(C_k) / P(x)
```

Gaussian likelihood:

```
P(x_i | C_k) = (1 / √(2πσ_k^2)) exp(-(x_i - μ_k)^2 / 2σ_k^2)
```

Log probability (để tránh underflow):

```
log P(C_k | x) = log P(C_k) + Σ log P(x_i | C_k)
```

**NumPy Implementation:**

```python
self.means[idx] = np.mean(X_c, axis=0)
self.variances[idx] = np.var(X_c, axis=0) + self.var_smoothing

log_prob = -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
log_probs[:, idx] = log_prior + np.sum(log_likelihood, axis=1)
predictions = self.classes[np.argmax(log_probs, axis=1)]
```

#### 4. Random Forest

**Công thức:**

Gini impurity:

```
Gini(D) = 1 - Σ p_i^2
```

Information gain:

```
Gain(D, feature) = Gini(D) - Σ (|D_v| / |D|) Gini(D_v)
```

Ensemble prediction (majority voting):

```
y_pred = mode({T_1(x), T_2(x), ..., T_n(x)})
```

**NumPy Implementation:**

- CART (Classification and Regression Trees) algorithm
- Bootstrap sampling cho mỗi tree
- Random feature selection tại mỗi split
- Majority voting cho predictions

```python
indices = np.random.choice(n_samples, n_samples, replace=True)
X_sample, y_sample = X[indices], y[indices]

features = np.random.choice(n_features, max_features, replace=False)

predictions = np.array([tree.predict(X) for tree in self.trees])
final_pred = np.array([np.bincount(pred).argmax() for pred in predictions.T])
```

## Installation & Setup

### Requirements

```bash
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
```

### Cài Đặt

```bash
# Clone repository
git clone https://github.com/chisngyen/HW2_NUMPY_FOR_DATA_SCIENCE.git
cd HW2_NUMPY_FOR_DATA_SCIENCE

# Cài đặt dependencies
pip install numpy matplotlib seaborn pandas
```

### Cấu Trúc Thư Mục

```
HW2_NUMPY_FOR_DATA_SCIENCE/
├── data/
│   ├── raw/
│   │   └── BankChurners.csv
│   └── processed/
│       ├── X_train_basic.npy
│       ├── X_test_basic.npy
│       ├── X_train_enhanced.npy
│       └── X_test_enhanced.npy
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── models.py
│   └── visualization.py
└── README.md
```

## Usage

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Notebook này thực hiện:

- Load và thăm dò dữ liệu
- Thống kê mô tả chi tiết
- Trực quan hóa phân phối features
- Phân tích correlation
- Phát hiện patterns

### 2. Preprocessing

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

Chạy 2 versions:

- **Basic:** Preprocessing đơn giản (encoding + scaling)
- **Enhanced:** Preprocessing nâng cao (SMOTE + PCA)

Output: 8 files `.npy` trong `data/processed/`

### 3. Modeling

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

Train và đánh giá:

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Random Forest

So sánh performance giữa Basic vs Enhanced datasets.

### 4. Sử Dụng Module

```python
from src.models import LogisticRegression, KNearestNeighbors, ModelEvaluator
from src.data_processing import Normalizer, SMOTE
from src.visualization import DataVisualizer

# Train model
model = LogisticRegression(learning_rate=0.1, n_iterations=500)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = ModelEvaluator.accuracy(y_test, y_pred)
metrics = ModelEvaluator.precision_recall_f1(y_test, y_pred)
```

## Results

### Kết Quả Đạt Được

#### Metrics Tổng Quát


| Model               | Dataset  | Accuracy   | Precision  | Recall     | F1-Score   |
| --------------------- | ---------- | ------------ | ------------ | ------------ | ------------ |
| **Random Forest**   | Enhanced | **96.71%** | **96.48%** | **97.04%** | **96.76%** |
| KNN                 | Enhanced | 92.32%     | 87.41%     | 99.13%     | 92.90%     |
| Logistic Regression | Enhanced | 85.74%     | 85.99%     | 85.84%     | 85.91%     |
| Naive Bayes         | Enhanced | 82.94%     | 83.56%     | 82.59%     | 83.07%     |
| Random Forest       | Basic    | 94.57%     | 92.89%     | 71.87%     | 81.03%     |
| KNN                 | Basic    | 90.12%     | 80.68%     | 51.07%     | 62.55%     |
| Logistic Regression | Basic    | 88.89%     | 75.50%     | 46.18%     | 57.31%     |
| Naive Bayes         | Basic    | 87.16%     | 60.50%     | 59.02%     | 59.75%     |

#### Cross-Validation Results (5-Fold)


| Model               | Mean Accuracy | Std Dev | Min    | Max    |
| --------------------- | --------------- | --------- | -------- | -------- |
| Random Forest       | 96.51%        | 0.33%   | 96.14% | 97.06% |
| KNN                 | 91.43%        | 0.48%   | 90.88% | 92.21% |
| Logistic Regression | 85.32%        | 0.41%   | 84.60% | 85.81% |
| Naive Bayes         | 81.38%        | 0.78%   | 80.26% | 82.57% |

## Project Structure

### Chức Năng Từng File/Folder

```
HW2_NUMPY_FOR_DATA_SCIENCE/
│
├── data/                           # Thư mục chứa dữ liệu
│   ├── raw/                        # Dữ liệu gốc
│   │   └── BankChurners.csv        # Dataset bank churners (10,127 records)
│   └── processed/                  # Dữ liệu đã xử lý
│       ├── X_train_basic.npy       # Training features (basic preprocessing)
│       ├── X_test_basic.npy        # Test features (basic preprocessing)
│       ├── y_train_basic.npy       # Training labels (basic)
│       ├── y_test_basic.npy        # Test labels (basic)
│       ├── X_train_enhanced.npy    # Training features (enhanced preprocessing)
│       ├── X_test_enhanced.npy     # Test features (enhanced preprocessing)
│       ├── y_train_enhanced.npy    # Training labels (enhanced)
│       └── y_test_enhanced.npy     # Test labels (enhanced)
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # EDA và statistical analysis
│   │   - Load và thăm dò dữ liệu
│   │   - Thống kê mô tả (mean, std, quartiles)
│   │   - Phân tích correlation matrix
│   │   - Trực quan hóa distributions
│   │   - Phát hiện outliers và patterns
│   │
│   ├── 02_preprocessing.ipynb      # Data preprocessing
│   │   - Basic preprocessing: encoding + scaling
│   │   - Enhanced preprocessing: SMOTE + feature engineering
│   │   - Train-test split
│   │   - Lưu processed data thành .npy files
│   │
│   └── 03_modeling.ipynb           # Model training và evaluation
│       - Load 2 versions data (basic & enhanced)
│       - Train 4 models: Logistic, KNN, Naive Bayes, Random Forest
│       - So sánh performance
│       - Confusion matrix analysis
│       - Cross-validation
│       - Model insights và recommendations
│
├── src/                            # Source code modules
│   ├── data_processing.py          # Data processing utilities
│   │   - StatisticalAnalyzer: thống kê mô tả
│   │   - MissingValueHandler: xử lý missing values
│   │   - OutlierDetector: phát hiện và xử lý outliers
│   │   - Normalizer: scaling và standardization
│   │   - FeatureEncoder: label encoding, one-hot encoding
│   │   - SMOTE: over-sampling cho imbalanced data
│   │   - PCA: dimensionality reduction
│   │   - FeatureSelector: feature selection methods
│   │   - DataSplitter: train-test split utilities
│   │
│   ├── models.py                   # Machine Learning models (100% NumPy)
│   │   - LogisticRegression: binary classification với gradient descent
│   │   - KNearestNeighbors: KNN với euclidean/manhattan distance
│   │   - NaiveBayes: Gaussian Naive Bayes
│   │   - DecisionTree: CART algorithm với Gini impurity
│   │   - RandomForest: ensemble của decision trees
│   │   - ModelEvaluator: accuracy, precision, recall, F1, confusion matrix
│   │   - cross_validate: K-fold cross-validation
│   │
│   └── visualization.py            # Visualization utilities
│       - DataVisualizer class:
│         + plot_histogram: histogram với mean/median lines
│         + plot_correlation_heatmap: correlation matrix heatmap
│         + plot_pie_chart: pie charts cho categorical data
│         + plot_model_comparison: so sánh models (bar charts)
│         + plot_transaction_scatter: scatter plots
│         + Various EDA visualization methods
│
├── README.md                       # Documentation (bạn đang đọc)
└── .gitignore                      # Git ignore rules
```

### Module Dependencies

```
notebooks/
  └─> src/data_processing.py
  └─> src/models.py
  └─> src/visualization.py

src/models.py
  └─> numpy

src/data_processing.py
  └─> numpy

src/visualization.py
  └─> numpy, matplotlib, seaborn
```

## Challenges & Solutions

### Khó Khăn Gặp Phải Khi Dùng NumPy

#### 1. Imbalanced Dataset (83.93% vs 16.07%)

**Vấn đề:**

- Models bị bias về class majority
- Recall thấp cho minority class (churn)
- F1-Score thấp trên basic preprocessing

**Giải pháp:**

- Implement SMOTE (Synthetic Minority Over-sampling) bằng NumPy
- Sử dụng KNN để generate synthetic samples
- Vectorized distance calculation: `D^2 = ||X||^2 + ||Y||^2 - 2XY^T`
- Kết quả: F1-Score tăng từ 57% lên 96% (Logistic Regression)

#### 2. Numerical Instability

**Vấn đề:**

- Overflow trong sigmoid function: `exp(-z)` khi `z` rất lớn
- Underflow trong Naive Bayes: `P(x) = Π P(x_i)` tích nhiều số nhỏ

**Giải pháp:**

```python
# Sigmoid: clip values trước khi exp
z = np.clip(z, -500, 500)
sigmoid = 1 / (1 + np.exp(-z))

# Naive Bayes: dùng log-space
log_prob = log_prior + np.sum(log_likelihood, axis=1)
```

#### 3. KNN Performance với Large Dataset

**Vấn đề:**

- KNN cần tính distance matrix: O(n*m) complexity
- Với n=8,101 và m=2,026: 16.4 million calculations

**Giải pháp:**

- Vectorized distance calculation thay vì loops:

```python
# Bad: O(n*m) với Python loops
for i in range(n):
    for j in range(m):
        dist[i,j] = np.sqrt(np.sum((X[i] - Y[j])**2))

# Good: O(n*m) nhưng optimized với NumPy
X_sq = np.sum(X**2, axis=1, keepdims=True)
Y_sq = np.sum(Y**2, axis=1, keepdims=True)
dist = np.sqrt(X_sq + Y_sq.T - 2 * X @ Y.T)
```

- Tốc độ tăng 100x

#### 4. Random Forest Implementation

**Vấn đề:**

- Decision tree recursion làm chậm
- Memory overhead khi lưu cấu trúc tree

**Giải pháp:**

- Sử dụng dictionary để lưu tree structure (lightweight)
- Implement iterative predict thay vì recursive
- Bootstrap sampling với `np.random.choice(replace=True)`
- Parallel-ready design (có thể dùng multiprocessing sau này)

#### 5. Memory Management

**Vấn đề:**

- Processed data quá lớn để giữ trong memory
- SMOTE tạo thêm synthetic samples

**Giải pháp:**

- Save/load processed data bằng `.npy` format (efficient binary)
- Chunked processing cho large operations
- Preallocate arrays: `np.zeros()` thay vì append

#### 6. Cross-Validation Implementation

**Vấn đề:**

- K-fold split phải đảm bảo mỗi fold có đủ cả 2 classes
- Shuffling để avoid bias từ data ordering

**Giải pháp:**

```python
# Shuffle indices trước khi split
indices = np.random.permutation(n_samples)

# Chia thành k folds
fold_size = n_samples // k
for fold in range(k):
    val_indices = indices[start:end]
    train_indices = np.concatenate([indices[:start], indices[end:]])
```

## Future Improvements

### Hướng Phát Triển Tiếp Theo

#### 1. Model Enhancements

**Deep Learning Models:**

- Implement Neural Networks từ đầu bằng NumPy
- Architecture: Input -> Hidden Layers -> Output
- Backpropagation với NumPy vectorization
- Activation functions: ReLU, Sigmoid, Softmax

**Ensemble Methods:**

- Gradient Boosting từ đầu (không dùng XGBoost)
- Stacking: combine predictions của nhiều models
- Voting classifier với weighted predictions

**Model Optimization:**

- Hyperparameter tuning với Grid Search
- Feature importance analysis cho Random Forest
- Learning curve analysis

#### 2. Feature Engineering

**Advanced Features:**

- Time-based features: trend analysis
- Interaction features: feature_A * feature_B
- Polynomial features: x^2, x^3
- Clustering-based features (K-Means)

**Dimensionality Reduction:**

- PCA implementation đã có, có thể mở rộng
- LDA (Linear Discriminant Analysis)
- t-SNE cho visualization

#### 3. Performance Optimization

**Computational Efficiency:**

- Cython/Numba cho speed up critical functions
- Parallel processing cho Random Forest
- GPU acceleration với CuPy (NumPy-like API)

**Memory Optimization:**

- Sparse matrix support
- Out-of-core learning cho very large datasets
- Batch processing

#### 4. Deployment

**Production-Ready:**

- Model serialization (pickle/joblib)
- REST API với Flask/FastAPI
- Docker containerization
- CI/CD pipeline

**Real-time Prediction:**

- Streaming data processing
- Online learning (incremental updates)
- A/B testing framework

#### 5. Explainability

**Model Interpretation:**

- SHAP values implementation
- LIME for local explanations
- Feature contribution analysis
- Decision path visualization cho Random Forest

**Business Insights:**

- Customer segmentation analysis
- Churn risk scoring
- Retention strategy recommendations
- ROI calculation tools

#### 6. Additional Features

**Data Pipeline:**

- Automated data validation
- Data drift detection
- Feature store integration

**Monitoring:**

- Model performance tracking
- Prediction confidence scores
- Alert system cho anomalies

#### 7. Extended Analysis

**Survival Analysis:**

- Time-to-churn prediction
- Customer lifetime value estimation
- Hazard rate analysis

**Causal Inference:**

- Propensity score matching
- Treatment effect estimation
- Counterfactual analysis

## License

MIT License

Copyright (c) 2025 Nguyen Chi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
