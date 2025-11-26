# Model Training Component - Inputs and Outputs Explanation

## Component Overview

The **model_training** component is a Kubeflow pipeline component that trains a Random Forest classifier on preprocessed training data and saves the trained model artifact for later use.

---

## Component Signature

```python
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
```

---

## Inputs

### 1. `train_data: Input[Dataset]`

**Type**: Input[Dataset]  
**Description**: Path to the preprocessed training dataset (CSV file)

**Purpose**:
- Contains the scaled and split training data from the preprocessing component
- Includes both features (X_train) and target variable (y_train)
- Data is already cleaned and normalized using StandardScaler

**Format**:
- CSV file with features as columns
- Last column is 'TARGET' (binary classification: 0 or 1)
- All features are scaled to have mean=0 and std=1

**Rationale**:
- Using `Input[Dataset]` type ensures proper data lineage tracking in Kubeflow
- The component receives the output from the preprocessing component
- Keeps data flow explicit and traceable

**Example**:
```
CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,TARGET
-0.42,0.28,-0.49,0.27,-0.14,0.41,-0.12,0.14,-0.98,-0.67,-1.46,0.44,-0.49,1
...
```

---

### 2. `n_estimators: int`

**Type**: Integer  
**Description**: Number of trees in the Random Forest

**Purpose**:
- Controls the ensemble size
- More trees generally improve performance but increase training time
- Typical values: 50-500

**Default/Recommended**: 100

**Rationale**:
- Hyperparameter that significantly affects model performance
- Making it an input allows experimentation without code changes
- Enables hyperparameter tuning in the pipeline
- Can be varied in different pipeline runs for optimization

**Impact**:
- Higher values: Better performance, longer training time, more memory
- Lower values: Faster training, potentially underfitting

---

### 3. `max_depth: int`

**Type**: Integer  
**Description**: Maximum depth of each decision tree

**Purpose**:
- Controls tree complexity and prevents overfitting
- Limits how deep each tree can grow
- Typical values: 5-30, or None for unlimited

**Default/Recommended**: 10

**Rationale**:
- Critical regularization parameter
- Prevents individual trees from memorizing training data
- Exposed as input for hyperparameter tuning
- Different datasets may require different depths

**Impact**:
- Higher values: More complex trees, risk of overfitting
- Lower values: Simpler trees, may underfit
- None: Trees grow until pure leaves (high overfitting risk)

---

### 4. `random_state: int`

**Type**: Integer  
**Description**: Random seed for reproducibility

**Purpose**:
- Ensures consistent results across multiple runs
- Controls randomness in:
  - Bootstrap sampling for each tree
  - Feature selection at each split
  - Random tie-breaking

**Default/Recommended**: 42

**Rationale**:
- **Reproducibility**: Critical for debugging and comparing experiments
- **Version Control**: Same seed = same model for same data
- **Experimentation**: Can change seed to test model stability
- **MLOps Best Practice**: Ensures deterministic pipeline behavior

**Impact**:
- Same seed: Identical models across runs
- Different seeds: Slightly different models (ensemble variance)

---

## Outputs

### 1. `model: Output[Model]`

**Type**: Output[Model]  
**Description**: Path where the trained model artifact is saved

**Purpose**:
- Stores the serialized Random Forest model
- Can be loaded by downstream components (evaluation, deployment)
- Persists the trained model for future use

**Format**:
- Serialized using `joblib` (`.pkl` file)
- Contains complete model state:
  - All trained decision trees
  - Feature names
  - Model parameters
  - Scikit-learn metadata

**Rationale**:
- Using `Output[Model]` type enables Kubeflow to track model artifacts
- Joblib is efficient for scikit-learn models (better than pickle)
- Model can be versioned and registered in MLflow Model Registry
- Enables model lineage tracking (which data/params produced this model)

**File Structure**:
```python
# Saved as: model.path + '.pkl'
# Can be loaded with: joblib.load(model_path)
```

---

### 2. `model_name: str` (Return Value)

**Type**: String  
**Description**: Identifier/name for the trained model

**Purpose**:
- Human-readable model identifier
- Used for logging and tracking
- Helps distinguish between different model versions

**Format**:
```python
f"random_forest_n{n_estimators}_d{max_depth}"
# Example: "random_forest_n100_d10"
```

**Rationale**:
- Encodes key hyperparameters in the name
- Makes it easy to identify model configuration
- Useful for MLflow experiment tracking
- Helps in model comparison and selection

---

### 3. `training_accuracy: float` (Return Value)

**Type**: Float (0.0 to 1.0)  
**Description**: Accuracy score on the training dataset

**Purpose**:
- Quick performance indicator
- Helps detect overfitting (if much higher than test accuracy)
- Validates that training succeeded

**Calculation**:
```python
training_accuracy = accuracy_score(y_train, y_train_pred)
```

**Rationale**:
- **Sanity Check**: Should be reasonably high (>0.7 for most datasets)
- **Overfitting Detection**: Compare with test accuracy
- **Pipeline Validation**: Confirms model learned something
- **Logging**: Automatically logged to MLflow for tracking

**Interpretation**:
- 0.95-1.0: Possible overfitting (especially if test << train)
- 0.7-0.95: Good training performance
- <0.7: Model may be underfitting or data quality issues

---

### 4. `num_trees: int` (Return Value)

**Type**: Integer  
**Description**: Number of trees actually trained in the forest

**Purpose**:
- Confirms the model was built with correct configuration
- Validation that n_estimators parameter was applied
- Useful for debugging and logging

**Value**:
```python
num_trees = n_estimators  # Should match input parameter
```

**Rationale**:
- **Verification**: Ensures model matches specification
- **Metadata**: Important for model documentation
- **Debugging**: Helps trace issues if model behaves unexpectedly
- **Logging**: Recorded in MLflow for experiment tracking

---

## Data Flow

```
Input: train_data (CSV) ──────────────┐
Input: n_estimators (100) ────────────┤
Input: max_depth (10) ────────────────├──> MODEL TRAINING ──┬──> Output: model (.pkl)
Input: random_state (42) ─────────────┘                      ├──> Return: model_name
                                                              ├──> Return: training_accuracy
                                                              └──> Return: num_trees
```

---

## Example Usage

```python
# In a Kubeflow pipeline
training_task = model_training(
    train_data=preprocessing_task.outputs['train_data'],
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Access outputs
model_artifact = training_task.outputs['model']
model_name = training_task.outputs['model_name']
train_acc = training_task.outputs['training_accuracy']
```

---

## Design Decisions

### Why Random Forest?

1. **Robust**: Handles non-linear relationships well
2. **No Feature Scaling Required**: But we scaled anyway for consistency
3. **Feature Importance**: Provides interpretability
4. **Ensemble Method**: Reduces overfitting through averaging
5. **Classification**: Perfect for binary classification (house price high/low)

### Why These Specific Inputs?

1. **n_estimators & max_depth**: Most impactful hyperparameters for Random Forest
2. **random_state**: Essential for reproducibility in MLOps
3. **train_data**: Natural output from preprocessing component

### Why These Specific Outputs?

1. **model**: Required for evaluation and deployment
2. **model_name**: Helps with experiment tracking
3. **training_accuracy**: Quick performance check
4. **num_trees**: Validation and metadata

---

## Integration with MLflow

While this is a Kubeflow component, it also integrates with MLflow:

```python
# Inside the component (optional)
import mlflow
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
mlflow.log_metric("training_accuracy", training_accuracy)
mlflow.sklearn.log_model(model, "random_forest_model")
```

This dual integration provides:
- **Kubeflow**: Pipeline orchestration and artifact tracking
- **MLflow**: Experiment tracking and model registry

---

## Summary

The **model_training** component is designed with:

✅ **Clear Inputs**: Data path + 3 hyperparameters  
✅ **Comprehensive Outputs**: Model artifact + 3 metadata values  
✅ **Flexibility**: Hyperparameters can be tuned without code changes  
✅ **Reproducibility**: Random state ensures consistent results  
✅ **Traceability**: All inputs/outputs tracked by Kubeflow  
✅ **Integration**: Works with both Kubeflow and MLflow  

This design follows MLOps best practices for modular, reusable, and traceable ML components.
