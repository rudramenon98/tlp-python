#!/bin/bash

# Script to run docker compose up for all docker-compose.yml files in the dockers directory
# Each subdirectory in dockers/ should contain a docker-compose.yml

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERS_DIR="${SCRIPT_DIR}/dockers"
BUILD_CONTEXT="${SCRIPT_DIR}/.."  # Build context is packages/ directory
# Get current directory
CURRENT_DIR="$(pwd)"

echo "Running docker compose up for all services in: ${DOCKERS_DIR}"
echo "Build context: ${BUILD_CONTEXT}"
echo ""

# Change to the build context directory (packages/) so compose file paths resolve correctly
cd "${BUILD_CONTEXT}"

# Iterate through each directory in dockers/
for docker_dir in "${DOCKERS_DIR}"/*; do
    # Check if it's a directory
    if [ -d "${docker_dir}" ]; then
        docker_name=$(basename "${docker_dir}")
        compose_file="${docker_dir}/docker-compose.yml"
        
        # Check if docker-compose.yml exists
        if [ -f "${compose_file}" ]; then
            echo "Starting Docker services for: ${docker_name}"
            echo "Compose file: ${compose_file}"

            

            echo "Running docker command:"
            echo "docker compose -f ${compose_file} --project-directory ${docker_dir} up -d"

            # Run docker compose up from the build context directory
            # Use -f to specify the compose file and --project-directory to set working directory
            docker compose -f "${compose_file}" --project-directory "${docker_dir}" up -d
            
            
            echo "Successfully started: ${docker_name}"
            echo ""
        else
            echo "Skipping ${docker_name}: No docker-compose.yml found"
            echo ""
        fi
    fi
done

echo "All Docker services started successfully!"

