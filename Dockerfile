FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["python", "pipeline.py"]
