"""
MLflow Pipeline Components for Boston Housing Dataset

This module contains reusable MLflow components for the ML pipeline.
Each component is designed to be tracked and logged with MLflow.
"""

import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os


class DataLoader:
    """Component for loading data from DVC-tracked sources"""
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_data(self):
        """Load Boston Housing dataset"""
        with mlflow.start_run(nested=True, run_name="data_loading"):
            mlflow.log_param("data_path", self.data_path)
            
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Log dataset info
            mlflow.log_param("num_samples", len(df))
            mlflow.log_param("num_features", len(df.columns) - 1)
            
            print(f"✓ Loaded data: {df.shape}")
            return df


class DataPreprocessor:
    """Component for data preprocessing and feature engineering"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def preprocess(self, df, target_column='MEDV'):
        """Preprocess data and split into train/test sets"""
        with mlflow.start_run(nested=True, run_name="data_preprocessing"):
            # Log parameters
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("target_column", target_column)
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Log preprocessing info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            print(f"✓ Preprocessed data - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


class ModelTrainer:
    """Component for training ML models"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
    
    def train(self, X_train, y_train, **model_params):
        """Train the model"""
        with mlflow.start_run(nested=True, run_name=f"model_training_{self.model_type}"):
            # Log model type
            mlflow.log_param("model_type", self.model_type)
            
            # Initialize model
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(**model_params)
            elif self.model_type == 'linear_regression':
                self.model = LinearRegression(**model_params)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Log model parameters
            for param, value in model_params.items():
                mlflow.log_param(param, value)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            print(f"✓ Trained {self.model_type} model")
            return self.model


class ModelEvaluator:
    """Component for evaluating trained models"""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, X_test, y_test, model_name="model"):
        """Evaluate model performance"""
        with mlflow.start_run(nested=True, run_name="model_evaluation"):
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Print results
            print(f"\n{'='*50}")
            print(f"Model Evaluation Results - {model_name}")
            print(f"{'='*50}")
            print(f"MSE:  {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"R²:   {r2:.4f}")
            print(f"{'='*50}\n")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }


class ModelRegistry:
    """Component for registering models to MLflow"""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
    
    def register(self, feature_names=None):
        """Register model to MLflow"""
        with mlflow.start_run(nested=True, run_name="model_registration"):
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=self.model_name
            )
            
            # Log feature names if provided
            if feature_names:
                mlflow.log_param("feature_names", feature_names)
            
            print(f"✓ Registered model: {self.model_name}")


if __name__ == "__main__":
    print("MLflow Pipeline Components Module")
    print("Import this module to use the components in your pipeline")
