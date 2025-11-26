"""
Kubeflow Pipeline Components for Boston Housing ML Pipeline

This module contains Kubeflow-compatible pipeline components that also integrate
with MLflow for experiment tracking. Each component is decorated with @component
and can be compiled to YAML for Kubeflow deployment.

Components:
1. data_extraction - Fetch data from DVC remote storage
2. data_preprocessing - Clean, scale, and split data
3. model_training - Train Random Forest classifier
4. model_evaluation - Evaluate model and save metrics
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple
import os


@component(
    base_image='python:3.9',
    packages_to_install=['dvc==3.48.0', 'pandas==2.1.4', 'mlflow==2.9.2']
)
def data_extraction(
    data_path: str,
    output_data: Output[Dataset]
) -> NamedTuple('Outputs', [('num_samples', int), ('num_features', int)]):
    """
    Data Extraction Component
    
    Fetches the versioned dataset from DVC remote storage and prepares it for processing.
    
    Args:
        data_path: Path to the DVC-tracked data file (e.g., 'data/raw/boston_housing.csv')
        output_data: Output path where extracted data will be saved
    
    Returns:
        num_samples: Number of samples in the dataset
        num_features: Number of features in the dataset
    """
    import pandas as pd
    import os
    from collections import namedtuple
    
    print("="*70)
    print("DATA EXTRACTION COMPONENT")
    print("="*70)
    
    # Check if data file exists
    if os.path.exists(data_path):
        print(f"✓ Loading data from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"⚠ Data file not found at {data_path}")
        print("Creating sample Boston Housing dataset...")
        
        # Create sample data if file doesn't exist
        import numpy as np
        np.random.seed(42)
        n_samples = 506
        
        df = pd.DataFrame({
            'CRIM': np.random.exponential(3.61, n_samples),
            'ZN': np.random.choice([0, 12.5, 25, 50, 75, 100], n_samples),
            'INDUS': np.random.gamma(2, 5.5, n_samples),
            'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),
            'NOX': np.random.beta(2, 2, n_samples) * 0.5 + 0.3,
            'RM': np.random.normal(6.28, 0.7, n_samples),
            'AGE': np.random.beta(2, 1, n_samples) * 100,
            'DIS': np.random.gamma(2, 2, n_samples),
            'RAD': np.random.choice(range(1, 25), n_samples),
            'TAX': np.random.normal(408, 168, n_samples),
            'PTRATIO': np.random.normal(18.5, 2.2, n_samples),
            'B': np.random.beta(10, 1, n_samples) * 400,
            'LSTAT': np.random.gamma(2, 6, n_samples),
        })
        
        # Generate target (for classification, we'll bin the values)
        medv = (
            50 + 8 * (df['RM'] - 6) - 2 * np.log1p(df['CRIM']) -
            0.5 * df['LSTAT'] + 5 * df['CHAS'] - 0.3 * df['AGE'] / 10 +
            np.random.normal(0, 5, n_samples)
        )
        medv = medv.clip(5, 50)
        
        # Convert to binary classification: High value (1) vs Low value (0)
        # Threshold at median
        df['TARGET'] = (medv > medv.median()).astype(int)
    
    # If no TARGET column exists, create one for classification
    if 'TARGET' not in df.columns and 'MEDV' in df.columns:
        median_price = df['MEDV'].median()
        df['TARGET'] = (df['MEDV'] > median_price).astype(int)
        df = df.drop(columns=['MEDV'])
    
    # Save extracted data
    os.makedirs(os.path.dirname(output_data.path), exist_ok=True)
    df.to_csv(output_data.path, index=False)
    
    num_samples = len(df)
    num_features = len(df.columns) - 1  # Exclude target
    
    print(f"\n✓ Data extracted successfully")
    print(f"  Samples: {num_samples}")
    print(f"  Features: {num_features}")
    print(f"  Saved to: {output_data.path}")
    print("="*70 + "\n")
    
    # Return named tuple
    outputs = namedtuple('Outputs', ['num_samples', 'num_features'])
    return outputs(num_samples, num_features)


@component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.1.4', 'scikit-learn==1.3.2', 'numpy==1.26.2', 'mlflow==2.9.2']
)
def data_preprocessing(
    input_data: Input[Dataset],
    test_size: float,
    random_state: int,
    train_data: Output[Dataset],
    test_data: Output[Dataset]
) -> NamedTuple('Outputs', [('train_samples', int), ('test_samples', int), ('num_features', int)]):
    """
    Data Preprocessing Component
    
    Handles data cleaning, feature scaling, and train/test splitting.
    
    Args:
        input_data: Input dataset from extraction component
        test_size: Proportion of data for test set (e.g., 0.2 for 20%)
        random_state: Random seed for reproducibility
        train_data: Output path for training dataset
        test_data: Output path for test dataset
    
    Returns:
        train_samples: Number of training samples
        test_samples: Number of test samples
        num_features: Number of features after preprocessing
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import namedtuple
    import os
    
    print("="*70)
    print("DATA PREPROCESSING COMPONENT")
    print("="*70)
    
    # Load data
    print(f"Loading data from: {input_data.path}")
    df = pd.read_csv(input_data.path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print("⚠ Handling missing values...")
        df = df.fillna(df.mean())
    
    # Separate features and target
    if 'TARGET' in df.columns:
        target_col = 'TARGET'
    elif 'MEDV' in df.columns:
        # Convert to classification
        median_val = df['MEDV'].median()
        df['TARGET'] = (df['MEDV'] > median_val).astype(int)
        df = df.drop(columns=['MEDV'])
        target_col = 'TARGET'
    else:
        raise ValueError("No target column found!")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\n✓ Features: {list(X.columns)}")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    print("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Add target back
    X_train_df['TARGET'] = y_train.values
    X_test_df['TARGET'] = y_test.values
    
    # Save processed data
    os.makedirs(os.path.dirname(train_data.path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data.path), exist_ok=True)
    
    X_train_df.to_csv(train_data.path, index=False)
    X_test_df.to_csv(test_data.path, index=False)
    
    train_samples = len(X_train_df)
    test_samples = len(X_test_df)
    num_features = len(X.columns)
    
    print(f"\n✓ Preprocessing complete")
    print(f"  Training samples: {train_samples}")
    print(f"  Test samples: {test_samples}")
    print(f"  Features: {num_features}")
    print(f"  Train data saved to: {train_data.path}")
    print(f"  Test data saved to: {test_data.path}")
    print("="*70 + "\n")
    
    outputs = namedtuple('Outputs', ['train_samples', 'test_samples', 'num_features'])
    return outputs(train_samples, test_samples, num_features)


@component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn==1.3.2', 'pandas==2.1.4', 'joblib==1.3.2', 'mlflow==2.9.2']
)
def model_training(
    train_data: Input[Dataset],
    n_estimators: int,
    max_depth: int,
    random_state: int,
    model: Output[Model]
) -> NamedTuple('Outputs', [('model_name', str), ('training_accuracy', float), ('num_trees', int)]):
    """
    Model Training Component
    
    Trains a Random Forest classifier on the training data and saves the model artifact.
    
    Args:
        train_data: Training dataset from preprocessing component
        n_estimators: Number of trees in the Random Forest
        max_depth: Maximum depth of each tree
        random_state: Random seed for reproducibility
        model: Output path where trained model will be saved
    
    Returns:
        model_name: Name/identifier of the trained model
        training_accuracy: Accuracy score on training data
        num_trees: Number of trees in the trained forest
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    import os
    from collections import namedtuple
    
    print("="*70)
    print("MODEL TRAINING COMPONENT")
    print("="*70)
    
    # Load training data
    print(f"Loading training data from: {train_data.path}")
    train_df = pd.read_csv(train_data.path)
    
    # Separate features and target
    X_train = train_df.drop(columns=['TARGET'])
    y_train = train_df['TARGET']
    
    print(f"✓ Training data loaded: {X_train.shape}")
    print(f"  Features: {list(X_train.columns)}")
    print(f"  Target classes: {sorted(y_train.unique())}")
    
    # Initialize Random Forest Classifier
    print(f"\nInitializing Random Forest Classifier...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  random_state: {random_state}")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Train model
    print("\nTraining model...")
    rf_model.fit(X_train, y_train)
    print("✓ Model training complete!")
    
    # Calculate training accuracy
    y_train_pred = rf_model.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    
    print(f"\nTraining Accuracy: {training_accuracy:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model.path), exist_ok=True)
    model_path = model.path + '.pkl'
    joblib.dump(rf_model, model_path)
    
    model_name = f"random_forest_n{n_estimators}_d{max_depth}"
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"  Model name: {model_name}")
    print("="*70 + "\n")
    
    outputs = namedtuple('Outputs', ['model_name', 'training_accuracy', 'num_trees'])
    return outputs(model_name, training_accuracy, n_estimators)


@component(
    base_image='python:3.9',
    packages_to_install=['scikit-learn==1.3.2', 'pandas==2.1.4', 'joblib==1.3.2', 'mlflow==2.9.2']
)
def model_evaluation(
    model: Input[Model],
    test_data: Input[Dataset],
    metrics: Output[Metrics]
) -> NamedTuple('Outputs', [('accuracy', float), ('f1_score', float), ('precision', float), ('recall', float)]):
    """
    Model Evaluation Component
    
    Evaluates the trained model on test data and saves comprehensive metrics.
    
    Args:
        model: Trained model from training component
        test_data: Test dataset from preprocessing component
        metrics: Output path where metrics JSON will be saved
    
    Returns:
        accuracy: Classification accuracy on test set
        f1_score: F1 score on test set
        precision: Precision score on test set
        recall: Recall score on test set
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    import json
    import os
    from collections import namedtuple
    
    print("="*70)
    print("MODEL EVALUATION COMPONENT")
    print("="*70)
    
    # Load model
    model_path = model.path + '.pkl'
    print(f"Loading model from: {model_path}")
    trained_model = joblib.load(model_path)
    print("✓ Model loaded successfully")
    
    # Load test data
    print(f"\nLoading test data from: {test_data.path}")
    test_df = pd.read_csv(test_data.path)
    
    # Separate features and target
    X_test = test_df.drop(columns=['TARGET'])
    y_test = test_df['TARGET']
    
    print(f"✓ Test data loaded: {X_test.shape}")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = trained_model.predict(X_test)
    
    # Calculate metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("="*70)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save metrics to JSON
    metrics_dict = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist(),
        'test_samples': len(y_test),
        'predictions': {
            'class_0': int((y_pred == 0).sum()),
            'class_1': int((y_pred == 1).sum())
        }
    }
    
    os.makedirs(os.path.dirname(metrics.path), exist_ok=True)
    with open(metrics.path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics.path}")
    print("="*70 + "\n")
    
    outputs = namedtuple('Outputs', ['accuracy', 'f1_score', 'precision', 'recall'])
    return outputs(accuracy, f1, precision, recall)


# Helper function to log components to MLflow (optional)
def log_component_to_mlflow(component_name, inputs, outputs):
    """
    Optional MLflow integration for component tracking
    """
    try:
        import mlflow
        with mlflow.start_run(run_name=f"component_{component_name}", nested=True):
            mlflow.log_params(inputs)
            mlflow.log_metrics(outputs)
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    print("Kubeflow Pipeline Components Module")
    print("=" * 70)
    print("\nAvailable Components:")
    print("1. data_extraction - Fetch data from DVC storage")
    print("2. data_preprocessing - Clean, scale, and split data")
    print("3. model_training - Train Random Forest classifier")
    print("4. model_evaluation - Evaluate model and save metrics")
    print("\nUse compile_components.py to generate YAML files")
    print("=" * 70)
