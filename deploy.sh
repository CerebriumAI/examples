#!/bin/bash
set -e

echo "Pulling latest code from Git..."
git pull

echo "Building and deploying Docker containers..."
docker compose pull
docker compose up -d --build

echo "Deployment complete. Containers are running."
