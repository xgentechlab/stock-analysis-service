#!/bin/bash
# Cloud Function deployment script

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"us-central1"}
FUNCTION_NAME=${FUNCTION_NAME:-"stock-selection-api"}

echo "Deploying Stock Selection API as Cloud Function"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Function: ${FUNCTION_NAME}"

# Deploy the function
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=stock_selection_api \
    --trigger=http \
    --allow-unauthenticated \
    --memory=1GB \
    --timeout=540s \
    --max-instances=10 \
    --set-env-vars ENVIRONMENT=production \
    --set-env-vars DEBUG=false \
    --requirements-file=requirements_functions.txt

echo "Function deployed successfully!"

# Get function URL
FUNCTION_URL=$(gcloud functions describe ${FUNCTION_NAME} --region=${REGION} --format="value(serviceConfig.uri)")
echo "Function URL: ${FUNCTION_URL}"

# Set up Cloud Scheduler jobs
echo "Setting up Cloud Scheduler jobs..."

# Daily selection job (runs at 9:15 AM IST = 3:45 AM UTC)
gcloud scheduler jobs create http daily-selection-job \
    --schedule="45 3 * * 1-5" \
    --uri="${FUNCTION_URL}/cron/run-selection" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Daily stock selection pipeline" \
    || echo "Daily selection job already exists"

# Position tracker job (runs every 15 minutes during market hours)
gcloud scheduler jobs create http position-tracker-job \
    --schedule="*/15 4-10 * * 1-5" \
    --uri="${FUNCTION_URL}/cron/run-tracker" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Position tracker job" \
    || echo "Position tracker job already exists"

echo "Cloud Scheduler jobs configured!"
echo "Setup complete. Function is available at: ${FUNCTION_URL}"
