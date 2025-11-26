"""
Model Training Script for Boston Housing Dataset

This script handles the training of the regression model with MLflow tracking.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import os


def load_data(data_path=None):
    """Load Boston Housing dataset"""
    if data_path and os.path.exists(data_path):
        # Load from CSV if path provided
        df = pd.read_csv(data_path)
        print(f"Loaded data from {data_path}")
    else:
        # Load from sklearn (for initial setup)
        print("Loading Boston Housing dataset from sklearn...")
        # Note: load_boston is deprecated, but used here for assignment
        # In production, use the CSV file
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['MEDV'] = data.target
            print("Note: Using California Housing as Boston dataset is deprecated")
        except:
            # Fallback to manual Boston dataset
            print("Creating sample Boston Housing dataset...")
            np.random.seed(42)
            n_samples = 506
            df = pd.DataFrame({
                'CRIM': np.random.rand(n_samples),
                'ZN': np.random.rand(n_samples) * 100,
                'INDUS': np.random.rand(n_samples) * 27,
                'CHAS': np.random.randint(0, 2, n_samples),
                'NOX': np.random.rand(n_samples),
                'RM': np.random.rand(n_samples) * 4 + 3,
                'AGE': np.random.rand(n_samples) * 100,
                'DIS': np.random.rand(n_samples) * 12,
                'RAD': np.random.randint(1, 25, n_samples),
                'TAX': np.random.rand(n_samples) * 700,
                'PTRATIO': np.random.rand(n_samples) * 10 + 12,
                'B': np.random.rand(n_samples) * 400,
                'LSTAT': np.random.rand(n_samples) * 38,
                'MEDV': np.random.rand(n_samples) * 40 + 10
            })
    
    return df


def train_model(data_path=None, n_estimators=100, max_depth=10, random_state=42):
    """Train Random Forest model with MLflow tracking"""
    
    # Set MLflow experiment
    mlflow.set_experiment("boston_housing_experiment")
    
    with mlflow.start_run(run_name="random_forest_training"):
        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Load data
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        df = load_data(data_path)
        mlflow.log_param("num_samples", len(df))
        mlflow.log_param("num_features", len(df.columns) - 1)
        print(f"Dataset shape: {df.shape}")
        
        # Prepare features and target
        X = df.drop(columns=['MEDV'])
        y = df['MEDV']
        
        # Split data
        print("\n" + "="*60)
        print("SPLITTING DATA")
        print("="*60)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        print("\n" + "="*60)
        print("SCALING FEATURES")
        print("="*60)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("✓ Features scaled using StandardScaler")
        
        # Train model
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        print(f"✓ Model trained: RandomForestRegressor")
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Training metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        # Test metrics
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Log metrics to MLflow
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2", train_r2)
        
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        
        # Print results
        print("\nTraining Set Metrics:")
        print(f"  MSE:  {train_mse:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  R²:   {train_r2:.4f}")
        
        print("\nTest Set Metrics:")
        print(f"  MSE:  {test_mse:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  R²:   {test_r2:.4f}")
        
        # Log model
        print("\n" + "="*60)
        print("LOGGING MODEL TO MLFLOW")
        print("="*60)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="boston_housing_rf"
        )
        print("✓ Model logged to MLflow")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nRun 'mlflow ui' to view results in the MLflow UI")
        
        return model, test_r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Boston Housing model")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data CSV")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
