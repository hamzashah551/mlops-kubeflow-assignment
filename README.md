# Boston Housing Prediction Pipeline (MLOps Assignment)

This repository contains an end-to-end MLOps pipeline for predicting Boston Housing prices. It demonstrates a hybrid approach using **Kubeflow Pipelines (KFP)** for pipeline definition and **MLflow** for local execution and experiment tracking.

## 🚀 Project Overview

The goal is to build a reproducible ML pipeline that predicts housing prices based on various features (crime rate, number of rooms, etc.).

**Key Features:**
*   **Data Versioning**: DVC is used to track the dataset.
*   **Pipeline Definition**: Components are defined using Kubeflow DSL (`@component`).
*   **Experiment Tracking**: MLflow tracks runs, parameters, metrics, and models.
*   **CI/CD**: GitHub Actions automates testing and pipeline compilation.
*   **Containerization**: Docker support for reproducible environments.

---

## 🛠️ Setup Instructions

### Prerequisites
*   Python 3.9+
*   Git
*   (Optional) Docker

### 1. Clone the Repository
```bash
git clone https://github.com/hamzashah551/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up DVC
The data is versioned with DVC. To pull the data from the remote storage:
```bash
dvc pull
```
*Note: The remote storage is configured as a local folder. Ensure you have access to the configured path.*

---

## 🏃‍♂️ Pipeline Walkthrough

### Option A: Run Locally with MLflow (Recommended)
This executes the pipeline steps (Extraction -> Preprocessing -> Training -> Evaluation) locally and logs results to MLflow.

1.  **Run the pipeline script:**
    ```bash
    python pipeline.py
    ```

2.  **View Results in MLflow UI:**
    ```bash
    mlflow ui
    ```
    Open [http://localhost:5000](http://localhost:5000) in your browser.

### Option B: Compile for Kubeflow
To generate the `pipeline.yaml` file for deployment to a Kubeflow cluster:

1.  **Run the compilation script:**
    ```bash
    python compile_components.py
    ```
    *This generates individual component YAMLs in `components/`.*

2.  **Compile the full pipeline:**
    ```bash
    python pipeline.py
    ```
    *This generates `pipeline.yaml` in the root directory.*

### Option C: CI/CD Pipeline
Every push to the `main` branch triggers a GitHub Actions workflow that:
1.  Sets up the environment.
2.  Compiles components to verify syntax.
3.  Runs the pipeline locally to verify functionality.

Check the **Actions** tab in GitHub to see the run history.

---

## 📂 Project Structure

```
mlops-kubeflow-assignment/
├── .github/workflows/      # CI/CD Pipeline (GitHub Actions)
├── components/             # Compiled Kubeflow Component YAMLs
├── data/                   # Data directory (tracked by DVC)
├── src/                    # Source code
│   ├── pipeline_components.py  # Component definitions (KFP + Python)
│   ├── model_training.py       # Standalone training script
│   └── ...
├── compile_components.py   # Script to compile components
├── pipeline.py             # Main pipeline definition & execution script
├── pipeline.yaml           # Compiled Kubeflow Pipeline
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker environment
└── README.md               # Project documentation
```

---

## 📊 Components

1.  **Data Extraction**: Loads data from DVC-tracked storage.
2.  **Data Preprocessing**: Cleans missing values, scales features, and splits into train/test sets.
3.  **Model Training**: Trains a Random Forest Regressor/Classifier.
4.  **Model Evaluation**: Calculates accuracy/R2 score and logs metrics.

---

## 📝 License
This project is part of an MLOps assignment.
