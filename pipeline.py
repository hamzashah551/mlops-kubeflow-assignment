"""
Main MLflow Pipeline for Boston Housing Prediction

This script orchestrates the complete ML pipeline using MLflow.
"""

import mlflow
import mlflow.sklearn
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline_components import (
    DataLoader,
    DataPreprocessor,
    ModelTrainer,
    ModelEvaluator,
    ModelRegistry
)


def run_pipeline(data_path="data/raw/boston_housing.csv", experiment_name="boston_housing_pipeline"):
    """
    Run the complete ML pipeline
    
    Args:
        data_path: Path to the dataset
        experiment_name: Name of the MLflow experiment
    """
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    print("\n" + "="*70)
    print("MLFLOW PIPELINE - BOSTON HOUSING PREDICTION")
    print("="*70)
    
    with mlflow.start_run(run_name="pipeline_run") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {experiment_name}\n")
        
        # Log pipeline parameters
        mlflow.log_param("pipeline_version", "1.0")
        mlflow.log_param("data_source", data_path)
        
        try:
            # Step 1: Load Data
            print("\n" + "-"*70)
            print("STEP 1: DATA LOADING")
            print("-"*70)
            loader = DataLoader(data_path)
            df = loader.load_data()
            
            # Step 2: Preprocess Data
            print("\n" + "-"*70)
            print("STEP 2: DATA PREPROCESSING")
            print("-"*70)
            preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
            X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess(df)
            
            # Step 3: Train Model
            print("\n" + "-"*70)
            print("STEP 3: MODEL TRAINING")
            print("-"*70)
            trainer = ModelTrainer(model_type='random_forest')
            model = trainer.train(
                X_train, y_train,
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Step 4: Evaluate Model
            print("\n" + "-"*70)
            print("STEP 4: MODEL EVALUATION")
            print("-"*70)
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X_test, y_test, model_name="Random Forest")
            
            # Step 5: Register Model
            print("\n" + "-"*70)
            print("STEP 5: MODEL REGISTRATION")
            print("-"*70)
            registry = ModelRegistry(model, model_name="boston_housing_model")
            registry.register(feature_names=feature_names)
            
            # Log final pipeline metrics
            mlflow.log_metric("pipeline_r2_score", metrics['r2'])
            mlflow.log_metric("pipeline_rmse", metrics['rmse'])
            
            print("\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nFinal Model Performance:")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  RMSE:     {metrics['rmse']:.4f}")
            print(f"\nView results: mlflow ui")
            print(f"Then navigate to: http://localhost:5000")
            print("="*70 + "\n")
            
            return model, metrics
            
        except FileNotFoundError:
            print(f"\n⚠ Warning: Data file not found at {data_path}")
            print("Running with sample data instead...\n")
            
            # Import and run the training script with sample data
            from src.model_training import train_model
            model, r2 = train_model(data_path=None)
            
            print("\n" + "="*70)
            print("PIPELINE COMPLETED WITH SAMPLE DATA")
            print("="*70)
            print(f"\nTo use real data:")
            print(f"  1. Add your dataset to: {data_path}")
            print(f"  2. Track it with DVC: dvc add {data_path}")
            print(f"  3. Run the pipeline again")
            print("="*70 + "\n")
            
            return model, {'r2': r2}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MLflow pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/boston_housing.csv",
        help="Path to dataset"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="boston_housing_pipeline",
        help="MLflow experiment name"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data_path,
        experiment_name=args.experiment_name
    )
