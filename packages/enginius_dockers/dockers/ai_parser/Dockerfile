###### Important ######
# This Dockerfile must be run with build context set to packages/ directory
# Use enginius-dockers/build_dockers.sh to build the docker image
######################

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


# Copy and install dependencies
COPY enginius_dockers/dockers/ai_parser/requirements.txt .
COPY enginius_dockers/dockers/ai_parser/main.py .
RUN pip install --no-cache-dir -r requirements.txt


COPY common_tools ./packages/common_tools
RUN pip install --no-cache-dir ./packages/common_tools

COPY database ./packages/database
RUN pip install --no-cache-dir ./packages/database

# Copy packages first (for local package installation)
COPY ai_doc_parser ./packages/ai_doc_parser
RUN pip install --no-cache-dir ./packages/ai_doc_parser


# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
