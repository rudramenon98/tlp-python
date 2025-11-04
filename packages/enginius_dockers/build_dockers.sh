#!/bin/bash

# Script to build all Docker images in the dockers directory
# Each subdirectory in dockers/ should contain a Dockerfile

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERS_DIR="${SCRIPT_DIR}/dockers"
BUILD_CONTEXT="${SCRIPT_DIR}/.."  # Build context is packages/ directory

echo "Building Docker images from: ${DOCKERS_DIR}"
echo "Build context: ${BUILD_CONTEXT}"
echo ""

# Iterate through each directory in dockers/
for docker_dir in "${DOCKERS_DIR}"/*; do
    # Check if it's a directory
    if [ -d "${docker_dir}" ]; then
        docker_name=$(basename "${docker_dir}")
        dockerfile_path="${docker_dir}/Dockerfile"
        
        # Check if Dockerfile exists
        if [ -f "${dockerfile_path}" ]; then
            echo "Building Docker image: ${docker_name}"
            echo "Dockerfile: ${dockerfile_path}"
            
            # Build the docker image
            # Build context is packages/, dockerfile is in the specific docker directory
            docker build \
                -f "${dockerfile_path}" \
                -t "${docker_name}:latest" \
                "${BUILD_CONTEXT}"

            # Remove existing container if it exists to avoid conflicts
            if docker ps -a --format '{{.Names}}' | grep -q "^${docker_name}$"; then
                echo "Removing existing container: ${docker_name}"
                docker rm -f "${docker_name}" 2>/dev/null || true
            fi

            # Run the docker image
            docker run -d --name "${docker_name}" "${docker_name}:latest"
            
            echo "Successfully built and started: ${docker_name}"
            echo ""
        else
            echo "Skipping ${docker_name}: No Dockerfile found"
            echo ""
        fi
    fi
done

echo "All Docker images built successfully!"

