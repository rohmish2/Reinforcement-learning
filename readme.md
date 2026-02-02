# ML-based Athletic Performance Analysis

A comprehensive machine learning project for analyzing and predicting athletic performance using multiple classification algorithms including K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Neural Networks.



##  Overview

This project implements advanced machine learning techniques to analyze athletic performance data and predict outcomes. The system uses multiple classification algorithms with extensive hyperparameter tuning through grid search to achieve optimal performance.

The analysis pipeline includes data preprocessing, feature engineering, model training with cross-validation, and comprehensive performance evaluation across three primary algorithms:
- K-Nearest Neighbors (KNN) with optimized k selection
- Support Vector Machines (SVM) with kernel optimization
- Neural Networks with architecture tuning

##  Features

- **Multiple ML Algorithms**: Implements KNN, SVM, and Neural Network classifiers
- **Automated Hyperparameter Tuning**: Grid search with parallel processing for optimal model parameters
- **Stratified Sampling**: Ensures balanced class representation in training data
- **Parallel Processing**: Maximized CPU core utilization for faster training
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Scalable Architecture**: Optimized for both small datasets and large-scale analysis
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Feature Engineering**: Advanced data preprocessing and transformation

## 🔧 Requirements

### Python Version
- Python 3.7 or higher

### Core Dependencies

```
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0
```


##  Usage

### Quick Start

Simply run the main Python script with the dataset:

```bash
python FinalTermProject.py
```

### Expected Runtime

- **Apple M3 Chip (8 cores)**: ~15 minutes

The runtime depends on:
- Number of CPU cores available
- Dataset size
- Grid search parameters
- Model complexity

### Step-by-Step Execution

1. **Data Loading**: The script automatically loads the CSV dataset
2. **Preprocessing**: Handles missing values, normalization, and encoding
3. **Training**: Executes grid search for each algorithm
4. **Evaluation**: Generates performance metrics and visualizations
5. **Results**: Outputs best parameters and model comparisons

##  Dataset

### Dataset Format

The project expects a CSV file containing athletic performance metrics from Open Powerlifting Dataset(https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database). Ensure your dataset includes:

- **Features**: Various athletic performance indicators (speed, strength, endurance, etc.)
- **Target Variable**: Classification labels for performance categories
- **Format**: CSV with headers in the first row

### Dataset Placement

Place your CSV dataset file in the project root directory alongside `FinalTermProject.py`.



##  Machine Learning Models

### 1. K-Nearest Neighbors (KNN)

**Algorithm**: Instance-based learning that classifies based on proximity to training examples.

**Key Features**:
- Distance-based classification
- No explicit training phase
- Sensitive to feature scaling

**Hyperparameters Tuned**:
- `n_neighbors`: Number of neighbors to consider
- `weights`: Uniform or distance-weighted
- `metric`: Distance metric (euclidean, manhattan, minkowski)
- `algorithm`: Tree algorithm for neighbors search

**Optimization Note**: To speed up KNN classification, you can increase the step count in the grid search:

```python
# Example: Faster search with larger steps
param_grid_knn = {
    'n_neighbors': range(1, 50, 5),  # Step of 5 instead of 1
    # ... other parameters
}
```

### 2. Support Vector Machine (SVM)

**Algorithm**: Finds optimal hyperplane for classification in high-dimensional space.

**Key Features**:
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors)
- Versatile kernel functions

**Hyperparameters Tuned**:
- `C`: Regularization parameter
- `kernel`: Kernel type (linear, rbf, poly, sigmoid)
- `gamma`: Kernel coefficient
- `degree`: Polynomial degree (for poly kernel)

**Optimization Note**: SVM training is computationally intensive. Training sample size is reduced to 10,000 samples with stratification to balance speed and accuracy. For full dataset training:

```python
# Remove sample size limitation
# Comment out or modify the sampling code
X_train_sample = X_train
y_train_sample = y_train
```

**Warning**: Full dataset training may take 2-3 hours depending on dataset size.

### 3. Neural Networks

**Algorithm**: Multi-layer perceptron with backpropagation.

**Key Features**:
- Non-linear decision boundaries
- Automatic feature learning
- Flexible architecture

**Hyperparameters Tuned**:
- `hidden_layer_sizes`: Network architecture
- `activation`: Activation function (relu, tanh, logistic)
- `solver`: Optimization algorithm (adam, sgd, lbfgs)
- `alpha`: L2 regularization parameter
- `learning_rate`: Learning rate schedule
- `max_iter`: Maximum iterations

**Optimization Note**: Like SVM, Neural Network training uses a reduced sample size (10,000) for faster execution. Adjust as needed for your computational resources.

##  Performance Optimization

### Parallel Processing

The project maximizes CPU core utilization:

```python
# Grid search uses all available cores
GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    n_jobs=-1,  # Uses all CPU cores
    verbose=2
)
```

### Memory Management

- **Stratified Sampling**: Maintains class balance while reducing memory footprint
- **Chunked Processing**: Large datasets can be processed in batches
- **Efficient Data Structures**: Uses NumPy arrays for computational efficiency

### Speed Optimization Tips

1. **Reduce Grid Search Space**: Narrow parameter ranges for faster convergence
2. **Decrease CV Folds**: Use 3-fold instead of 5-fold cross-validation
3. **Sample Data**: Use stratified sampling for initial experiments
4. **Parallelize**: Ensure `n_jobs=-1` in all GridSearchCV calls
5. **Cache Results**: Save trained models to avoid retraining

### System Requirements

**Minimum**:
- 4 GB RAM
- 2 CPU cores
- 1 GB free disk space

**Recommended**:
- 8 GB RAM or higher
- 4+ CPU cores
- 5 GB free disk space
- SSD for faster I/O

## 📈 Results

### Performance Metrics

The system evaluates models using:

- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate per class
- **Recall**: Sensitivity per class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Cross-Validation Score**: Mean and standard deviation across folds

### Output Files

The project generates:

1. **Model Performance Report**: Detailed metrics for each algorithm
2. **Comparison Visualizations**: Bar charts comparing model performance
3. **Confusion Matrices**: Visual representation of classification results
4. **Feature Importance Plots**: (if applicable)
5. **Training History**: Loss and accuracy curves for neural networks
6. **Best Parameters**: Optimal hyperparameters for each model

### Sample Output

```
=== K-NEAREST NEIGHBORS ===
Best Parameters: {'n_neighbors': 7, 'weights': 'distance', 'metric': 'euclidean'}
Best Cross-Validation Score: 0.8547
Test Accuracy: 0.8612
Precision: 0.8598
Recall: 0.8612
F1-Score: 0.8601

=== SUPPORT VECTOR MACHINE ===
Best Parameters: {'C': 10, 'kernel': 'rbf', 'gamma': 0.001}
Best Cross-Validation Score: 0.8723
Test Accuracy: 0.8756
...
```

## 📁 Project Structure

```
ML-based-Athletic-Performance-Analysis-/
│
├── FinalTermProject.py          # Main script
├── FTP Report.pdf               # Project report and documentation
├── readme.md                    # This file
├── requirements.txt             # Python dependencies (if available)
│
├── data/                        # Dataset directory
│   └── athletic_data.csv        # Athletic performance dataset
│
├── models/                      # Saved trained models
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   └── nn_model.pkl
│
├── results/                     # Output visualizations and reports
│   ├── confusion_matrices/
│   ├── performance_plots/
│   └── comparison_charts/
│
└── notebooks/                   # Jupyter notebooks (if available)
    └── exploratory_analysis.ipynb
```

## 🔬 Technical Details

### Data Preprocessing Pipeline

1. **Missing Value Handling**
   - Imputation strategies for numerical and categorical features
   - Removal of samples with excessive missing data

2. **Feature Scaling**
   - StandardScaler for zero mean and unit variance
   - MinMaxScaler for bounded features
   - Robust scaling for outlier-heavy features

3. **Encoding**
   - One-hot encoding for nominal categorical variables
   - Label encoding for ordinal variables

4. **Feature Engineering**
   - Polynomial features for non-linear relationships
   - Interaction terms between key features
   - Domain-specific feature creation

### Model Training Workflow

```python
# Pseudocode for training workflow
1. Load and preprocess data
2. Split into train/test sets (stratified)
3. For each model:
   a. Define parameter grid
   b. Initialize GridSearchCV
   c. Fit on training data
   d. Evaluate on test data
   e. Save best model
4. Compare models
5. Generate visualizations
6. Export results
```

### Cross-Validation Strategy

- **Method**: Stratified K-Fold (k=5)
- **Purpose**: Ensure robust performance estimates
- **Benefits**: Reduces overfitting, provides confidence intervals

### Grid Search Parameters

#### KNN Grid
```python
{
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}
```

#### SVM Grid
```python
{
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'degree': [2, 3, 4]  # for poly kernel
}
```

#### Neural Network Grid
```python
{
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 500, 1000]
}
```

##  Future Enhancements

### Planned Features

- [ ] **Deep Learning Models**: Integration of TensorFlow/PyTorch for advanced architectures
- [ ] **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost
- [ ] **Feature Selection**: Automated feature importance analysis and selection
- [ ] **Real-time Prediction**: REST API for live performance prediction
- [ ] **Web Dashboard**: Interactive visualization dashboard



##  Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue describing the bug
2. **Suggest Features**: Propose new features or improvements
3. **Submit Pull Requests**: Fix bugs or add features
4. **Improve Documentation**: Enhance README or code comments
5. **Share Datasets**: Provide additional athletic performance datasets


##  Authors

- **rohmish2** - *Initial work* - [GitHub Profile](https://github.com/rohmish2)

##  Acknowledgments

- Thanks to the scikit-learn community for excellent ML tools
- Athletic performance dataset contributors
- Open-source community for continuous support



##  References

### Machine Learning
- Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011
- Hastie et al., "The Elements of Statistical Learning", Springer 2009
- Bishop, "Pattern Recognition and Machine Learning", Springer 2006


---
