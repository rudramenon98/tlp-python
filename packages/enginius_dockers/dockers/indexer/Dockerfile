# Use official Python image
FROM python:3.11-slim

# Install build tools
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y -q build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY static-indexer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY static-indexer/ .

# Copy the app module
COPY sub_modules/tlp-java/tlp-scripts/app ./app

# Expose port
EXPOSE 8800

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8800", "--log-level", "debug"]

