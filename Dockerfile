# Use Python 3.11 as base image (stable and widely supported)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create outputs directory
RUN mkdir -p outputs

# Expose volume for data and outputs
VOLUME ["/app/data", "/app/outputs"]

# Default command
CMD ["python", "main.py", "--file", "data/source_data.xls", "--output", "outputs/"]
