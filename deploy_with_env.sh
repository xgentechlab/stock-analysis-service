#!/bin/bash
# Cloud Run deployment script with environment variables

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"trading-agent-469211"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"stock-selection-api"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Deploying Stock Selection API to Cloud Run with environment variables"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Build and push image
echo "Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

echo "Deploying to Cloud Run with environment variables..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8000 \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars ENVIRONMENT=production \
    --set-env-vars DEBUG=false \
    --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID} \
    --set-env-vars FIRESTORE_PROJECT_ID=${PROJECT_ID} \
    --set-env-vars FIRESTORE_DATABASE_ID=stock-data \
    --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json

echo "Deployment completed!"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format "value(status.url)")
echo "Service URL: ${SERVICE_URL}"

# Test the service
echo "Testing the service..."
curl -s "${SERVICE_URL}/health" | jq . || echo "Health check response (raw):"
curl -s "${SERVICE_URL}/health"

echo ""
echo "To set additional environment variables (like OpenAI API key), run:"
echo "gcloud run services update ${SERVICE_NAME} --region=${REGION} --set-env-vars OPENAI_API_KEY=your-key-here"
echo ""
echo "Setup complete. Service is available at: ${SERVICE_URL}"

