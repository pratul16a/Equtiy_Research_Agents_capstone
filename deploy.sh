#!/bin/bash
# Deploy Indian Equity Analyst to Google Cloud Run
# Usage:
#   bash deploy.sh          # Standard deploy (4 vCPU, safe rates)
#   bash deploy.sh demo     # Demo mode (8 vCPU, aggressive parallelism)

set -e

MODE="${1:-standard}"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="asia-south1"
SERVICE_NAME="indian-equity-analyst"
IMAGE_NAME="asia-south1-docker.pkg.dev/${PROJECT_ID}/indian-equity/analyst"

# ── Performance profiles ──
if [ "$MODE" = "demo" ]; then
    CPU=16
    MEMORY=16Gi
    YFINANCE_RPS=100
    SCREENER_WORKERS=128
    SCREENER_BATCH_PARALLEL=6
    echo "=== DEMO MODE: 16 vCPU / 16GB / Max throughput ==="
else
    CPU=4
    MEMORY=4Gi
    YFINANCE_RPS=30
    SCREENER_WORKERS=32
    SCREENER_BATCH_PARALLEL=5
    echo "=== STANDARD MODE ==="
fi

echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "CPU: ${CPU} | Memory: ${MEMORY} | RPS: ${YFINANCE_RPS} | Workers: ${SCREENER_WORKERS}"
echo ""

# Step 1: Build
echo "[1/2] Building container image..."
gcloud builds submit --tag "${IMAGE_NAME}" .

# Step 2: Deploy
echo "[2/2] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory "${MEMORY}" \
    --cpu "${CPU}" \
    --timeout 900 \
    --concurrency 1 \
    --set-env-vars "OPENROUTER_API_KEY=${OPENROUTER_API_KEY},LLM_MODEL=${LLM_MODEL:-openai/gpt-4o},YFINANCE_RPS=${YFINANCE_RPS},SCREENER_WORKERS=${SCREENER_WORKERS},SCREENER_BATCH_PARALLEL=${SCREENER_BATCH_PARALLEL}"

echo ""
echo "=== Deployed! ==="
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format "value(status.url)"
