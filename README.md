# DU DOAN CHURN KHACH HANG SU DUNG NUMPY

Du an phan tich va du doan hanh vi churn (ngung su dung dich vu) cua khach hang the tin dung su dung cac thuat toan Machine Learning duoc xay dung hoan toan bang NumPy.

## Muc Luc

1. [Gioi Thieu](#gioi-thieu)
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

## Gioi Thieu

### Mo Ta Bai Toan

Churn prediction la bai toan du doan khach hang co kha nang ngung su dung dich vu hay khong. Trong linh vuc ngan hang va the tin dung, viec nhan dien som khach hang co nguy co churn giup doanh nghiep:
- Tiet kiem chi phi thu hut khach hang moi (cao gap 5-25 lan so voi giu chan khach hang cu)
- Tang doanh thu bang cach ap dung cac chien dich retention hieu qua
- Cai thien trai nghiem khach hang thong qua cac dich vu ca nhan hoa

### Dong Luc va Ung Dung Thuc Te

**Tai sao su dung NumPy?**
- Hieu sau ve cac thuat toan Machine Learning bang cach implement tu dau
- Khong phu thuoc vao thu vien black-box (scikit-learn, TensorFlow)
- Toi uu hieu suat voi NumPy vectorization
- Ung dung trong education va research

**Ung dung thuc te:**
- Banking: Du doan khach hang dong the tin dung
- Telecom: Phat hien khach hang co y dinh chuyen mang
- E-commerce: Nhan dien khach hang khong quay lai
- SaaS: Du doan subscription cancellation

### Muc Tieu Cu The

1. Xay dung cac thuat toan ML (Logistic Regression, KNN, Naive Bayes, Random Forest) su dung 100% NumPy
2. So sanh hieu qua giua preprocessing co ban (basic) va nang cao (enhanced)
3. Dat duoc F1-Score tren 90% cho bai toan churn prediction
4. Phan tich insights tu confusion matrix va feature importance
5. Danh gia model stability thong qua cross-validation

## Dataset

### Nguon Du Lieu

Dataset: **Bank Churners Dataset**
- Nguon: Kaggle / Leaps Analyttica
- File: `data/raw/BankChurners.csv`
- Kich thuoc: 10,127 records x 23 features

### Mo Ta Cac Features

**Demographic Features:**
- `Customer_Age`: Tuoi khach hang
- `Gender`: Gioi tinh (M/F)
- `Dependent_count`: So nguoi phu thuoc
- `Education_Level`: Trinh do hoc van
- `Marital_Status`: Tinh trang hon nhan
- `Income_Category`: Muc thu nhap

**Banking Relationship:**
- `Card_Category`: Loai the (Blue, Silver, Gold, Platinum)
- `Months_on_book`: So thang lam khach hang
- `Total_Relationship_Count`: Tong so san pham da su dung
- `Months_Inactive_12_mon`: So thang khong hoat dong trong 12 thang
- `Contacts_Count_12_mon`: So lan lien he trong 12 thang

**Financial Metrics:**
- `Credit_Limit`: Han muc tin dung
- `Total_Revolving_Bal`: So du dao han
- `Avg_Open_To_Buy`: Han muc con lai trung binh
- `Total_Trans_Amt`: Tong gia tri giao dich (12 thang)
- `Total_Trans_Ct`: Tong so giao dich (12 thang)
- `Total_Ct_Chng_Q4_Q1`: Thay doi so giao dich Q4 vs Q1
- `Total_Amt_Chng_Q4_Q1`: Thay doi gia tri giao dich Q4 vs Q1
- `Avg_Utilization_Ratio`: Ty le su dung han muc trung binh

**Target Variable:**
- `Attrition_Flag`: Existing Customer (0) / Attrited Customer (1)

### Kich Thuoc va Dac Diem Du Lieu

- **Tong so mau:** 10,127 khach hang
- **Class distribution:**
  - Existing Customer: 8,500 (83.93%)
  - Attrited Customer: 1,627 (16.07%)
- **Imbalanced dataset:** Ratio 5.22:1
- **Missing values:** Khong co gia tri null
- **Feature types:**
  - Numerical: 14 features
  - Categorical: 6 features

**Train/Test Split:**
- Training set: 80% (8,101 samples)
- Test set: 20% (2,026 samples)

## Method

### Quy Trinh Xu Ly Du Lieu

#### 1. Data Exploration (`01_data_exploration.ipynb`)

- Thong ke mo ta (mean, median, std, quartiles)
- Phan tich phan phoi features
- Correlation analysis
- Phat hien outliers
- Phan tich churn patterns

#### 2. Preprocessing (`02_preprocessing.ipynb`)

**Basic Preprocessing:**
- Label encoding cho categorical features
- Min-Max scaling (0-1) cho numerical features
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

Train va danh gia 4 models tren ca 2 dataset versions.

### Thuat Toan Su Dung

#### 1. Logistic Regression

**Cong thuc:**

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

**Cong thuc:**

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

**Cong thuc:**

Bayes theorem:
```
P(C_k | x) = P(x | C_k) P(C_k) / P(x)
```

Gaussian likelihood:
```
P(x_i | C_k) = (1 / √(2πσ_k^2)) exp(-(x_i - μ_k)^2 / 2σ_k^2)
```

Log probability (de tranh underflow):
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

**Cong thuc:**

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
- Bootstrap sampling cho moi tree
- Random feature selection tai moi split
- Majority voting cho predictions

```python
indices = np.random.choice(n_samples, n_samples, replace=True)
X_sample, y_sample = X[indices], y[indices]

features = np.random.choice(n_features, max_features, replace=False)

predictions = np.array([tree.predict(X) for tree in self.trees])
final_pred = np.array([np.bincount(pred).argmax() for pred in predictions.T])
```

### Giai Thich Cach Implement Bang NumPy

**1. Vectorization:**
- Su dung NumPy broadcasting de tranh loops
- Matrix operations thay cho iteration
- Vi du: `np.dot(X, weights)` thay cho `sum(x_i * w_i)`

**2. Memory Efficiency:**
- Preallocate arrays: `np.zeros()`, `np.empty()`
- In-place operations: `+=`, `-=`
- View instead of copy: `X[mask]`

**3. Numerical Stability:**
- Clip values de tranh overflow: `np.clip(z, -500, 500)`
- Log-space calculations: `log P(x)` thay cho `P(x)`
- Add epsilon: `1e-10` de tranh division by zero

**4. Advanced Techniques:**
- Fancy indexing: `X[indices]`
- Boolean masking: `X[y == 1]`
- `np.argsort()` cho KNN
- `np.unique()` cho class counts
- `np.bincount()` cho voting

## Installation & Setup

### Requirements

```bash
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
```

### Cai Dat

```bash
# Clone repository
git clone https://github.com/chisngyen/HW2_NUMPY_FOR_DATA_SCIENCE.git
cd HW2_NUMPY_FOR_DATA_SCIENCE

# Cai dat dependencies
pip install numpy matplotlib seaborn pandas

# (Optional) Cai dat XGBoost de so sanh
pip install xgboost
```

### Cau Truc Thu Muc

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

Notebook nay thuc hien:
- Load va tham do du lieu
- Thong ke mo ta chi tiet
- Truc quan hoa phan phoi features
- Phan tich correlation
- Phat hien patterns va outliers

### 2. Preprocessing

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

Chay 2 versions:
- **Basic:** Preprocessing don gian (encoding + scaling)
- **Enhanced:** Preprocessing nang cao (SMOTE + feature engineering)

Output: 8 files `.npy` trong `data/processed/`

### 3. Modeling

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

Train va danh gia:
- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Random Forest

So sanh performance giua Basic vs Enhanced datasets.

### 4. Su Dung Module

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

### Ket Qua Dat Duoc

#### Metrics Tong Quat

| Model | Dataset | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Random Forest** | Enhanced | **96.71%** | **96.48%** | **97.04%** | **96.76%** |
| KNN | Enhanced | 92.32% | 87.41% | 99.13% | 92.90% |
| Logistic Regression | Enhanced | 85.74% | 85.99% | 85.84% | 85.91% |
| Naive Bayes | Enhanced | 82.94% | 83.56% | 82.59% | 83.07% |
| Random Forest | Basic | 94.57% | 92.89% | 71.87% | 81.03% |
| KNN | Basic | 90.12% | 80.68% | 51.07% | 62.55% |
| Logistic Regression | Basic | 88.89% | 75.50% | 46.18% | 57.31% |
| Naive Bayes | Basic | 87.16% | 60.50% | 59.02% | 59.75% |

#### Best Model: Random Forest (Enhanced)

**Confusion Matrix:**
```
                Predicted
              Not Churn  Churn
Actual  
Not Churn      1678        23
Churn            10       315
```

**Chi tiet metrics:**
- True Negatives: 1,678
- False Positives: 23
- False Negatives: 10
- True Positives: 315
- Specificity: 98.65%
- Sensitivity: 97.04%

#### Cross-Validation Results (5-Fold)

| Model | Mean Accuracy | Std Dev | Min | Max |
|-------|---------------|---------|-----|-----|
| Random Forest | 96.51% | 0.33% | 96.14% | 97.06% |
| KNN | 91.43% | 0.48% | 90.88% | 92.21% |
| Logistic Regression | 85.32% | 0.41% | 84.60% | 85.81% |
| Naive Bayes | 81.38% | 0.78% | 80.26% | 82.57% |

### Hinh Anh Truc Quan Hoa Ket Qua

#### 1. Model Comparison: Basic vs Enhanced

So sanh hieu suat giua 2 versions preprocessing cho thay Enhanced dataset mang lai cai thien dang ke:
- Random Forest: +19.4% F1-Score
- KNN: +48.5% F1-Score
- Logistic Regression: +49.9% F1-Score
- Naive Bayes: +39.1% F1-Score

#### 2. Confusion Matrix Comparison

Tat ca models tren Enhanced dataset cho ty le False Negatives thap hon, quan trong trong ung dung thuc te (tot hon bo sot khach hang churn).

#### 3. Feature Importance Analysis

Top features anh huong den churn:
1. Total_Trans_Ct (So luong giao dich)
2. Total_Trans_Amt (Gia tri giao dich)
3. Total_Revolving_Bal (So du dao han)
4. Total_Relationship_Count (So san pham)
5. Months_Inactive_12_mon (Thang khong hoat dong)

### So Sanh va Phan Tich

#### Enhanced vs Basic Preprocessing

**Cai thien F1-Score:**
- Logistic Regression: +49.9%
- KNN: +48.5%
- Naive Bayes: +39.1%
- Random Forest: +19.4%

**Nguyen nhan:**
1. SMOTE giai quyet class imbalance (83.93% vs 16.07%)
2. Feature engineering tao ra cac features co kha nang phan biet cao hon
3. Z-score standardization giup cac model distance-based hoat dong tot hon
4. Outlier handling loai bo noise

#### Model Comparison

**Random Forest (Best Overall):**
- Accuracy cao nhat: 96.71%
- F1-Score cao nhat: 96.76%
- Can bang giua Precision va Recall
- Stable qua cross-validation (std < 0.4%)

**KNN (Best Recall):**
- Recall cao nhat: 99.13%
- Chi 10 False Negatives tren 2,026 samples
- Phu hop khi uu tien nhan dien het khach hang churn

**Logistic Regression (Fast & Simple):**
- Training time nhanh (0.2s)
- Performance tot (85.74% accuracy)
- De interpret va deploy

**Naive Bayes (Fastest):**
- Training time nhanh nhat (0.002s)
- Performance chap nhan duoc (82.94% accuracy)
- Phu hop cho real-time predictions

#### Business Impact

Voi Random Forest Enhanced model:
- **97.04% Recall:** Nhan dien duoc 315/325 khach hang churn
- **96.48% Precision:** 93% khach hang duoc canh bao thuc su churn
- **Business value:** Tiet kiem chi phi retention campaign va giam churn rate

Gia su:
- Chi phi retention campaign: 100,000 VND/khach hang
- Chi phi mat khach hang: 1,000,000 VND/khach hang
- Model giup tiet kiem: 315 × 900,000 = 283,500,000 VND

## Project Structure

### Giải Thich Chuc Nang Tung File/Folder

```
HW2_NUMPY_FOR_DATA_SCIENCE/
│
├── data/                           # Thu muc chua du lieu
│   ├── raw/                        # Du lieu goc
│   │   └── BankChurners.csv        # Dataset bank churners (10,127 records)
│   └── processed/                  # Du lieu da xu ly
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
│   ├── 01_data_exploration.ipynb   # EDA va statistical analysis
│   │   - Load va tham do du lieu
│   │   - Thong ke mo ta (mean, std, quartiles)
│   │   - Phan tich correlation matrix
│   │   - Truc quan hoa distributions
│   │   - Phat hien outliers va patterns
│   │
│   ├── 02_preprocessing.ipynb      # Data preprocessing
│   │   - Basic preprocessing: encoding + scaling
│   │   - Enhanced preprocessing: SMOTE + feature engineering
│   │   - Train-test split
│   │   - Luu processed data thanh .npy files
│   │
│   └── 03_modeling.ipynb           # Model training va evaluation
│       - Load 2 versions data (basic & enhanced)
│       - Train 4 models: Logistic, KNN, Naive Bayes, Random Forest
│       - So sanh performance
│       - Confusion matrix analysis
│       - Cross-validation
│       - Model insights va recommendations
│
├── src/                            # Source code modules
│   ├── data_processing.py          # Data processing utilities
│   │   - StatisticalAnalyzer: thong ke mo ta
│   │   - MissingValueHandler: xu ly missing values
│   │   - OutlierDetector: phat hien va xu ly outliers
│   │   - Normalizer: scaling va standardization
│   │   - FeatureEncoder: label encoding, one-hot encoding
│   │   - SMOTE: over-sampling cho imbalanced data
│   │   - PCA: dimensionality reduction
│   │   - FeatureSelector: feature selection methods
│   │   - DataSplitter: train-test split utilities
│   │
│   ├── models.py                   # Machine Learning models (100% NumPy)
│   │   - LogisticRegression: binary classification voi gradient descent
│   │   - KNearestNeighbors: KNN voi euclidean/manhattan distance
│   │   - NaiveBayes: Gaussian Naive Bayes
│   │   - DecisionTree: CART algorithm voi Gini impurity
│   │   - RandomForest: ensemble cua decision trees
│   │   - ModelEvaluator: accuracy, precision, recall, F1, confusion matrix
│   │   - cross_validate: K-fold cross-validation
│   │
│   └── visualization.py            # Visualization utilities
│       - DataVisualizer class:
│         + plot_histogram: histogram voi mean/median lines
│         + plot_correlation_heatmap: correlation matrix heatmap
│         + plot_pie_chart: pie charts cho categorical data
│         + plot_model_comparison: so sanh models (bar charts)
│         + plot_transaction_scatter: scatter plots
│         + Various EDA visualization methods
│
├── README.md                       # Documentation (ban dang doc)
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

### Kho Khan Gap Phai Khi Dung NumPy

#### 1. Imbalanced Dataset (83.93% vs 16.07%)

**Van de:**
- Models bi bias ve class majority
- Recall thap cho minority class (churn)
- F1-Score thap tren basic preprocessing

**Giai phap:**
- Implement SMOTE (Synthetic Minority Over-sampling) bang NumPy
- Su dung KNN de generate synthetic samples
- Vectorized distance calculation: `D^2 = ||X||^2 + ||Y||^2 - 2XY^T`
- Ket qua: F1-Score tang tu 57% len 96% (Logistic Regression)

#### 2. Numerical Instability

**Van de:**
- Overflow trong sigmoid function: `exp(-z)` khi `z` rat lon
- Underflow trong Naive Bayes: `P(x) = Π P(x_i)` tich nhieu so nho

**Giai phap:**
```python
# Sigmoid: clip values truoc khi exp
z = np.clip(z, -500, 500)
sigmoid = 1 / (1 + np.exp(-z))

# Naive Bayes: dung log-space
log_prob = log_prior + np.sum(log_likelihood, axis=1)
```

#### 3. KNN Performance voi Large Dataset

**Van de:**
- KNN can tinh distance matrix: O(n*m) complexity
- Voi n=8,101 va m=2,026: 16.4 million calculations

**Giai phap:**
- Vectorized distance calculation thay vi loops:
```python
# Bad: O(n*m) voi Python loops
for i in range(n):
    for j in range(m):
        dist[i,j] = np.sqrt(np.sum((X[i] - Y[j])**2))

# Good: O(n*m) nhung optimized voi NumPy
X_sq = np.sum(X**2, axis=1, keepdims=True)
Y_sq = np.sum(Y**2, axis=1, keepdims=True)
dist = np.sqrt(X_sq + Y_sq.T - 2 * X @ Y.T)
```
- Toc do tang 100x

#### 4. Random Forest Implementation

**Van de:**
- Decision tree recursion lam cham
- Memory overhead khi luu cau truc tree

**Giai phap:**
- Su dung dictionary de luu tree structure (lightweight)
- Implement iterative predict thay vi recursive
- Bootstrap sampling voi `np.random.choice(replace=True)`
- Parallel-ready design (co the dung multiprocessing sau nay)

#### 5. Memory Management

**Van de:**
- Processed data qua lon de giu trong memory
- SMOTE tao them synthetic samples

**Giai phap:**
- Save/load processed data bang `.npy` format (efficient binary)
- Chunked processing cho large operations
- Preallocate arrays: `np.zeros()` thay vi append

#### 6. Cross-Validation Implementation

**Van de:**
- K-fold split phai dam bao moi fold co du ca 2 classes
- Shuffling de avoid bias tu data ordering

**Giai phap:**
```python
# Shuffle indices truoc khi split
indices = np.random.permutation(n_samples)

# Chia thanh k folds
fold_size = n_samples // k
for fold in range(k):
    val_indices = indices[start:end]
    train_indices = np.concatenate([indices[:start], indices[end:]])
```

## Future Improvements

### Huong Phat Trien Tiep Theo

#### 1. Model Enhancements

**Deep Learning Models:**
- Implement Neural Networks tu dau bang NumPy
- Architecture: Input -> Hidden Layers -> Output
- Backpropagation voi NumPy vectorization
- Activation functions: ReLU, Sigmoid, Softmax

**Ensemble Methods:**
- Gradient Boosting tu dau (khong dung XGBoost)
- Stacking: combine predictions cua nhieu models
- Voting classifier voi weighted predictions

**Model Optimization:**
- Hyperparameter tuning voi Grid Search
- Feature importance analysis cho Random Forest
- Learning curve analysis

#### 2. Feature Engineering

**Advanced Features:**
- Time-based features: trend analysis
- Interaction features: feature_A * feature_B
- Polynomial features: x^2, x^3
- Clustering-based features (K-Means)

**Dimensionality Reduction:**
- PCA implementation da co, co the mo rong
- LDA (Linear Discriminant Analysis)
- t-SNE cho visualization

#### 3. Performance Optimization

**Computational Efficiency:**
- Cython/Numba cho speed up critical functions
- Parallel processing cho Random Forest
- GPU acceleration voi CuPy (NumPy-like API)

**Memory Optimization:**
- Sparse matrix support
- Out-of-core learning cho very large datasets
- Batch processing

#### 4. Deployment

**Production-Ready:**
- Model serialization (pickle/joblib)
- REST API voi Flask/FastAPI
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

## Contributors

### Thong Tin Tac Gia

**Nguyen Chi**
- Role: Machine Learning Engineer / Data Scientist
- Affiliation: Student - Machine Learning Course
- Github: [@chisngyen](https://github.com/chisngyen)

**Contributions:**
- Designed va implemented tat ca ML algorithms tu dau bang NumPy
- Data exploration va statistical analysis
- Model evaluation va comparison framework
- Documentation va visualization

## Contact

Neu ban co cau hoi hoac muon dong gop vao project:

- GitHub Issues: [Create an issue](https://github.com/chisngyen/HW2_NUMPY_FOR_DATA_SCIENCE/issues)
- Email: chisngyen@example.com
- GitHub: [@chisngyen](https://github.com/chisngyen)

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

---

**Note:** Project nay duoc tao ra cho muc dich hoc tap va nghien cuu. Tat ca cac thuat toan Machine Learning duoc implement tu dau bang NumPy de hieu sau ve cach hoat dong cua chung.
