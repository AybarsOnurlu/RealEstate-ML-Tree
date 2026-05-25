#!/bin/bash
echo "======================================================="
echo "Starting Real Estate ML Tree (Explainable PropTech)..."
echo "======================================================="
echo ""

docker compose up -d --build

echo ""
echo "======================================================="
echo "Application is starting up!"
echo "It will download data and train the model on first boot."
echo "Please wait about ~30 seconds for the healthcheck to pass."
echo ""
echo "You can access the UI at: http://localhost:8080/"
echo "API Documentation at: http://localhost:8080/docs"
echo "======================================================="
