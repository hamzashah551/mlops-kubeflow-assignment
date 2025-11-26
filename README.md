# MLOps Assignment - MLflow & DVC Pipeline

This project demonstrates MLOps best practices using MLflow for experiment tracking and model management, DVC for data versioning, and Docker for containerization.

## 🎯 Project Overview

This assignment implements a complete ML pipeline for the Boston Housing dataset with:
- **MLflow**: Experiment tracking, model registry, and pipeline orchestration
- **DVC**: Data version control and management
- **Docker**: Containerized components
- **GitHub Actions**: CI/CD automation

## 📁 Project Structure

```
mlops-kubeflow-assignment/
├── data/
│   ├── raw/              # Raw data tracked by DVC
│   └── processed/        # Processed data
├── src/
│   ├── pipeline_components.py  # MLflow component definitions
│   └── model_training.py       # Training script
├── components/           # Compiled components
├── pipeline.py          # Main MLflow pipeline
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container image definition
├── .github/
│   └── workflows/      # GitHub Actions workflows
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- DVC
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hamzashah551/mlops-kubeflow-assignment.git
   cd mlops-kubeflow-assignment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull data from DVC**
   ```bash
   dvc pull
   ```

## 📊 Usage

### Run the Pipeline

```bash
python pipeline.py
```

### View MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### DVC Commands

```bash
# Check data status
dvc status

# Pull data from remote
dvc pull

# Push data to remote
dvc push
```

## 🧪 Testing

```bash
pytest tests/
```

## 📝 Assignment Tasks

- [x] Task 1: Project Initialization and Data Versioning
- [ ] Task 2: TBD
- [ ] Task 3: TBD

## 👤 Author

**Hamza Shah**
- GitHub: [@hamzashah551](https://github.com/hamzashah551)

## 📄 License

This project is for educational purposes as part of an MLOps assignment.
