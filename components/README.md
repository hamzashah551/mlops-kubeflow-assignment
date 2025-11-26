# Kubeflow Pipeline Components

This directory contains the compiled YAML definitions for the ML pipeline components.

## Components

### 1. data_extraction.yaml
- **Purpose**: Fetch versioned dataset from DVC remote storage
- **Inputs**: data_path (string)
- **Outputs**: output_data (Dataset), num_samples (int), num_features (int)

### 2. data_preprocessing.yaml
- **Purpose**: Clean, scale, and split data into train/test sets
- **Inputs**: input_data (Dataset), test_size (float), random_state (int)
- **Outputs**: train_data (Dataset), test_data (Dataset), train_samples (int), test_samples (int), num_features (int)

### 3. model_training.yaml
- **Purpose**: Train Random Forest classifier and save model artifact
- **Inputs**: train_data (Dataset), n_estimators (int), max_depth (int), random_state (int)
- **Outputs**: model (Model), model_name (string), training_accuracy (float), num_trees (int)

### 4. model_evaluation.yaml
- **Purpose**: Evaluate trained model on test set and save metrics
- **Inputs**: model (Model), test_data (Dataset)
- **Outputs**: metrics (Metrics), accuracy (float), f1_score (float), precision (float), recall (float)

## Usage

These components are defined in `src/pipeline_components.py` with the `@component` decorator from `kfp.dsl`.

The YAML files serve as documentation and can be used for Kubeflow deployment.

## Implementation

The actual implementation is in Python with MLflow integration for experiment tracking.

See `src/pipeline_components.py` for the complete code.
