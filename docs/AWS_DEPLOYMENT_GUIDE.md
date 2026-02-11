# AWS Deployment Guide - Sepsis Classification System

## Overview

This guide covers deploying the Sepsis Classification API (both Hybrid ML and GenAI pathways) to AWS using **ECR + App Runner**.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Deployment Steps (ECR + App Runner)](#deployment-steps-ecr--app-runner)
4. [Environment Variables Configuration](#environment-variables-configuration)
5. [Post-Deployment Verification](#post-deployment-verification)
6. [Integration with Backend Systems](#integration-with-backend-systems)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Cost Estimation](#cost-estimation)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. AWS Account Setup

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- IAM user/role with these permissions:
  - `AmazonEC2ContainerRegistryFullAccess`
  - `AWSAppRunnerFullAccess`
  - `CloudWatchLogsFullAccess`

### 2. Local Requirements

```bash
# Check AWS CLI
aws --version

# Check Docker
docker --version

# Configure AWS CLI (if not done)
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1)
```

### 3. Docker Image Ready

Ensure your local Docker image is built and tested:

```bash
cd /path/to/Alternative
docker-compose build
docker-compose up -d
curl http://localhost:8000/health  # Should return healthy
docker-compose down
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AWS DEPLOYMENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐  │
│   │   Local     │     │    AWS      │     │      AWS App Runner         │  │
│   │   Docker    │────►│    ECR      │────►│  (Auto-scaling container)   │  │
│   │   Image     │push │  Registry   │     │                             │  │
│   └─────────────┘     └─────────────┘     └──────────────┬──────────────┘  │
│                                                          │                  │
│                                                          │ HTTPS            │
│                                                          ▼                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    https://xxxxx.awsapprunner.com                    │  │
│   │                                                                      │  │
│   │  Endpoints:                                                          │  │
│   │  ├── GET  /health                 (Health Check)                     │  │
│   │  ├── GET  /docs                   (Swagger UI)                       │  │
│   │  ├── POST /classify               (Hybrid ML)                        │  │
│   │  ├── POST /classify-batch         (Hybrid ML Batch)                  │  │
│   │  ├── POST /genai-classify         (GenAI + LLM)                      │  │
│   │  ├── POST /genai-classify-batch   (GenAI Batch)                      │  │
│   │  └── GET  /genai-health           (GenAI Health)                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              │ API Call                                     │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Azure OpenAI (GPT-4o)                           │  │
│   │                 https://app-tx-peer.openai.azure.com                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Steps (ECR + App Runner)

**Pros:** Simple, managed, auto-scaling, no server management  
**Cons:** Slightly higher cost for low traffic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ECR + APP RUNNER FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   YOUR LOCAL MACHINE                                                        │
│   ┌─────────────────┐                                                       │
│   │ Docker Image    │                                                       │
│   │ sepsis-classifier│                                                      │
│   └────────┬────────┘                                                       │
│            │ docker push                                                    │
│            ▼                                                                │
│   AWS ECR (Elastic Container Registry)                                      │
│   ┌─────────────────────────────────────┐                                   │
│   │ xxxx.dkr.ecr.region.amazonaws.com/  │                                   │
│   │ sepsis-classifier:latest            │                                   │
│   └────────┬────────────────────────────┘                                   │
│            │ Auto-deploy                                                    │
│            ▼                                                                │
│   AWS APP RUNNER                                                            │
│   ┌─────────────────────────────────────┐                                   │
│   │ • Auto-scales                       │                                   │
│   │ • HTTPS enabled                     │                                   │
│   │ • URL: https://xxxxx.awsapprunner.com│                                  │
│   └────────┬────────────────────────────┘                                   │
│            │                                                                │
│            ▼                                                                │
│   COLLEAGUE'S BACKEND / MOBILE APP                                          │
│   ┌─────────────────────────────────────┐                                   │
│   │ Calls:                              │                                   │
│   │ • POST /classify         (Hybrid ML)│                                   │
│   │ • POST /genai-classify   (GenAI)    │                                   │
│   │ • GET  /health                      │                                   │
│   └─────────────────────────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Create ECR Repository

```bash
# Set variables
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME="sepsis-classifier"

# Create ECR repository
aws ecr create-repository \
    --repository-name $ECR_REPO_NAME \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true

# Output will show repository URI like:
# xxxxxxxxxxxx.dkr.ecr.us-east-1.amazonaws.com/sepsis-classifier
```

### Step 2: Authenticate Docker to ECR

```bash
# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

### Step 3: Tag and Push Image

```bash
# Tag the image
docker tag alternative-sepsis-api:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

# Also tag with version
docker tag alternative-sepsis-api:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:v2.0.0

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:v2.0.0
```

### Step 4: Create App Runner Service (Console)

1. Go to **AWS Console** → **App Runner** → **Create service**

2. **Source Configuration:**
   - Source: `Container registry`
   - Provider: `Amazon ECR`
   - Container image URI: `<account-id>.dkr.ecr.us-east-1.amazonaws.com/sepsis-classifier:latest`
   - ECR access role: Create new or use existing

3. **Deployment Settings:**
   - Deployment trigger: `Automatic` (deploys on new image push)
   - ECR access role: `Create new service role`

4. **Service Settings:**
   - Service name: `sepsis-classification-api`
   - CPU: `1 vCPU`
   - Memory: `2 GB`
   - Port: `8000`

5. **Environment Variables:** (Add all of these)
   ```
   API_KEY=sepsis_secret_key_2024
   AZURE_OPENAI_API_KEY=6fa83d52a5c0492e8690c009d09492d5
   AZURE_OPENAI_ENDPOINT=https://app-tx-peer.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-06-01
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_MODEL=gpt-4o
   PYTHONUNBUFFERED=1
   ```

6. **Auto Scaling:**
   - Min instances: `1`
   - Max instances: `10`
   - Max concurrency: `100`

7. **Health Check:**
   - Protocol: `HTTP`
   - Path: `/health`
   - Interval: `10 seconds`
   - Timeout: `5 seconds`
   - Healthy threshold: `1`
   - Unhealthy threshold: `5`

8. Click **Create & deploy**

### Step 4 Alternative: Create App Runner Service (CLI)

```bash
# Create App Runner service using CLI
aws apprunner create-service \
    --service-name sepsis-classification-api \
    --source-configuration '{
        "ImageRepository": {
            "ImageIdentifier": "'$AWS_ACCOUNT_ID'.dkr.ecr.'$AWS_REGION'.amazonaws.com/sepsis-classifier:latest",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8000",
                "RuntimeEnvironmentVariables": {
                    "API_KEY": "sepsis_secret_key_2024",
                    "AZURE_OPENAI_API_KEY": "6fa83d52a5c0492e8690c009d09492d5",
                    "AZURE_OPENAI_ENDPOINT": "https://app-tx-peer.openai.azure.com/",
                    "AZURE_OPENAI_API_VERSION": "2024-06-01",
                    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o",
                    "PYTHONUNBUFFERED": "1"
                }
            }
        },
        "AutoDeploymentsEnabled": true,
        "AuthenticationConfiguration": {
            "AccessRoleArn": "arn:aws:iam::'$AWS_ACCOUNT_ID':role/AppRunnerECRAccessRole"
        }
    }' \
    --instance-configuration '{
        "Cpu": "1024",
        "Memory": "2048"
    }' \
    --health-check-configuration '{
        "Protocol": "HTTP",
        "Path": "/health",
        "Interval": 10,
        "Timeout": 5,
        "HealthyThreshold": 1,
        "UnhealthyThreshold": 5
    }' \
    --region $AWS_REGION
```

### Step 5: Get Service URL

After deployment completes (5-10 minutes):

```bash
# Get service URL
aws apprunner list-services --region $AWS_REGION

# Or from console: App Runner → Services → sepsis-classification-api
# URL format: https://xxxxxxxx.us-east-1.awsapprunner.com
```

---

## Environment Variables Configuration

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `API_KEY` | Authentication key for API access | `sepsis_secret_key_2024` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | `6fa83d52a5c0...` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | `https://app-tx-peer.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-06-01` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name | `gpt-4o` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONUNBUFFERED` | Disable output buffering | `1` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

### Security Best Practice: Use AWS Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
    --name sepsis-api-secrets \
    --secret-string '{
        "API_KEY": "sepsis_secret_key_2024",
        "AZURE_OPENAI_API_KEY": "your-actual-key"
    }'

# Reference in App Runner using IAM role
```

---

## Post-Deployment Verification

### 1. Health Check

```bash
# Replace with your actual URL
API_URL="https://xxxxxxxx.us-east-1.awsapprunner.com"

# Basic health
curl $API_URL/health

# Expected response:
# {"status":"healthy","timestamp":"2026-02-10T..."}
```

### 2. Swagger Documentation

Open in browser:
```
https://xxxxxxxx.us-east-1.awsapprunner.com/docs
```

### 3. Test Hybrid ML Endpoint

```bash
curl -X POST "$API_URL/classify" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: sepsis_secret_key_2024" \
    -d '{
        "HR": 110,
        "SBP": 95,
        "DBP": 60,
        "Temp": 38.5,
        "Resp": 22,
        "O2Sat": 94,
        "WBC": 15.5,
        "Lactate": 2.5,
        "Creatinine": 1.8,
        "Age": 65,
        "Gender": 1
    }'
```

### 4. Test GenAI Endpoint

```bash
curl -X POST "$API_URL/genai-classify" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: sepsis_secret_key_2024" \
    -d '{
        "patient_id": "TEST001",
        "vitals": {
            "HR": 118,
            "SBP": 88,
            "Temp": 38.9,
            "Lactate": 3.2,
            "WBC": 18.5
        },
        "notes": "Patient appears lethargic, skin mottled on lower extremities"
    }'
```

### 5. Check GenAI Health

```bash
curl $API_URL/genai-health

# Expected: {"status":"available","pipeline_version":"1.0.0",...}
```

---

## Integration with Backend Systems

### What to Share with Your Colleague

| Item | Value |
|------|-------|
| **Base URL** | `https://xxxxxxxx.us-east-1.awsapprunner.com` |
| **API Key Header** | `X-API-Key: sepsis_secret_key_2024` |
| **Swagger Docs** | `{base_url}/docs` |
| **OpenAPI Spec** | `{base_url}/openapi.json` |

### Available Endpoints

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ENDPOINT                      │ METHOD │ AUTH │ DESCRIPTION               │
├────────────────────────────────────────────────────────────────────────────┤
│ /                             │ GET    │ No   │ API info and version      │
│ /health                       │ GET    │ No   │ Health check              │
│ /docs                         │ GET    │ No   │ Swagger UI                │
│ /openapi.json                 │ GET    │ No   │ OpenAPI specification     │
├────────────────────────────────────────────────────────────────────────────┤
│ /classify                     │ POST   │ Yes  │ Hybrid ML classification  │
│ /classify-batch               │ POST   │ Yes  │ Batch ML classification   │
│ /config/current               │ GET    │ Yes  │ Current ML config         │
│ /reload-config                │ POST   │ Yes  │ Reload ML config          │
├────────────────────────────────────────────────────────────────────────────┤
│ /genai-classify               │ POST   │ Yes  │ GenAI classification      │
│ /genai-classify-batch         │ POST   │ Yes  │ Batch GenAI classification│
│ /genai-health                 │ GET    │ No   │ GenAI pipeline health     │
│ /genai-guardrail/thresholds   │ GET    │ Yes  │ Current guardrail config  │
│ /genai-guardrail/reload       │ POST   │ Yes  │ Reload guardrail config   │
├────────────────────────────────────────────────────────────────────────────┤
│ /performance                  │ GET    │ Yes  │ Performance metrics       │
└────────────────────────────────────────────────────────────────────────────┘
```

### Sample Integration Code (Python)

```python
import requests

class SepsisAPIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    
    def classify_hybrid(self, patient_data):
        """Use Hybrid ML pathway"""
        response = requests.post(
            f"{self.base_url}/classify",
            json=patient_data,
            headers=self.headers
        )
        return response.json()
    
    def classify_genai(self, patient_id, vitals, notes=""):
        """Use GenAI pathway"""
        response = requests.post(
            f"{self.base_url}/genai-classify",
            json={
                "patient_id": patient_id,
                "vitals": vitals,
                "notes": notes
            },
            headers=self.headers
        )
        return response.json()

# Usage
client = SepsisAPIClient(
    base_url="https://xxxxxxxx.us-east-1.awsapprunner.com",
    api_key="sepsis_secret_key_2024"
)

# Hybrid ML
result = client.classify_hybrid({"HR": 110, "SBP": 95, "Temp": 38.5, ...})

# GenAI
result = client.classify_genai("P001", {"HR": 110, "SBP": 88}, "Patient lethargic")
```

---

## Monitoring and Logging

### CloudWatch Logs

App Runner automatically sends logs to CloudWatch:

```bash
# View logs
aws logs tail /aws/apprunner/sepsis-classification-api/service --follow
```

### CloudWatch Alarms (Recommended)

Create alarms for:
- High latency (P95 > 5s)
- Error rate > 5%
- CPU utilization > 80%
- Memory utilization > 80%

```bash
# Example: High error rate alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "SepsisAPI-HighErrorRate" \
    --alarm-description "Alert when error rate exceeds 5%" \
    --metric-name "5xxErrors" \
    --namespace "AWS/AppRunner" \
    --statistic Sum \
    --period 300 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=ServiceName,Value=sepsis-classification-api \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:alerts-topic
```

---

## Cost Estimation

| Resource | Specification | Monthly Cost (est.) |
|----------|---------------|---------------------|
| Compute | 1 vCPU, 2GB RAM | $30-50 |
| Provisioned instances | 1 minimum | Included |
| Requests | Up to 1M/month | Included |
| Data transfer | 10 GB | ~$1 |
| **Total** | | **~$35-55/month** |

---

## Troubleshooting

### Issue: Container fails to start

```bash
# Check App Runner logs
aws logs tail /aws/apprunner/sepsis-classification-api/service --since 1h

# Common causes:
# - Missing environment variables
# - Port mismatch (must be 8000)
# - Health check failing
```

### Issue: 503 Service Unavailable

```bash
# Check if service is running
aws apprunner describe-service --service-arn <service-arn>

# Check health endpoint directly
curl -v https://xxxxx.awsapprunner.com/health
```

### Issue: GenAI endpoint returns 503

```bash
# Check Azure OpenAI connectivity
curl https://xxxxx.awsapprunner.com/genai-health

# Verify environment variables are set correctly
# Check Azure OpenAI endpoint and key
```

### Issue: Slow cold starts

- Increase minimum instances to 1+ (always warm)
- Consider provisioned concurrency for critical workloads

### Issue: Authentication failures

```bash
# Verify API key header
curl -v -H "X-API-Key: sepsis_secret_key_2024" https://xxxxx.awsapprunner.com/health

# Check for typos in header name (must be X-API-Key)
```

---

## Quick Reference: Deployment Checklist

```
□ AWS CLI configured with proper credentials
□ Docker image built and tested locally
□ ECR repository created
□ Image pushed to ECR
□ App Runner service created with:
  □ Correct image URI
  □ Port set to 8000
  □ All environment variables configured
  □ Health check path set to /health
  □ Auto-scaling configured
□ Service deployed successfully
□ Health check passing
□ Swagger UI accessible
□ Test classification endpoints
□ Share credentials with backend team
□ Set up CloudWatch alarms
```

---

*Last Updated: February 2026*
*Version: 2.0*
