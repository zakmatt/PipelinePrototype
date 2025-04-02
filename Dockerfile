FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create necessary directories
RUN mkdir -p data/01_raw data/02_intermediate data/03_primary data/04_model data/05_model_input data/06_reporting data/07_pipeline data/08_reporting logs

# Set the entrypoint
ENTRYPOINT ["python", "run.py"]

# By default run the full pipeline
CMD []
