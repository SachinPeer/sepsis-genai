# AWS Bedrock Claude 3.5 Setup Guide

## Current Status

Your AWS account requires the Anthropic use case form to be submitted before Claude models can be accessed.

## Setup Steps

### Step 1: Submit Anthropic Use Case Form (Required)

1. Open AWS Console: https://console.aws.amazon.com/bedrock/
2. Select region: **us-east-1** (N. Virginia)
3. In the left sidebar, click **Model access**
4. Find **Anthropic** models and click **Request model access** or **Modify model access**
5. Fill out the use case form with:

```
Company Name: MedBacon Healthcare AI
Company Website: https://medbacon.com (or your actual website)
Industry: Healthcare
Use Case: Clinical Decision Support

Use Case Description:
GenAI-powered clinical decision support system for predicting sepsis risk 
in hospitalized patients. The system analyzes vital signs trends and nursing 
notes to identify silent sepsis before clinical deterioration, enabling 
early intervention and improved patient outcomes.

Expected Monthly Volume: 10,000-50,000 requests
Country: United States
```

6. Accept the Anthropic Acceptable Use Policy (AUP)
7. Submit the form

**Approval Time:** Usually instant to 24 hours

### Step 2: Verify Access

After approval, run this test:

```bash
cd sepsis-genai
python -c "
from genai_inference_service import BedrockClaudeProvider
provider = BedrockClaudeProvider()
health = provider.health_check()
print(health)
"
```

Expected output:
```
{'status': 'healthy', 'provider': 'aws_bedrock', 'model': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0', 'region': 'us-east-1'}
```

### Step 3: Run Model Comparison

Once Bedrock is healthy:

```bash
python compare_models.py --output model_comparison_results.json
```

## Configuration

Your `.env` should include:

```bash
# Provider selection
LLM_PROVIDER=bedrock  # or "azure" for Azure OpenAI

# AWS Bedrock Config
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Resource Tags Applied

| Resource | Tags |
|----------|------|
| ECR: sepsis-genai | Project=SepsisEarlyWarning, Purpose=GenAI Sepsis Classification Model, Environment=Production |
| ECR: sepsis-classifier | Project=SepsisEarlyWarning, Purpose=Hybrid ML Sepsis Classification Model, Environment=Production |

## Pricing Comparison

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3.5 Sonnet v2 | $3.00 | $15.00 |
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Azure GPT-4o | $5.00 | $15.00 |

## Troubleshooting

**Error: "Model use case details have not been submitted"**
- Complete Step 1 above to submit the use case form

**Error: "Access denied"**
- Ensure your IAM user/role has `bedrock:InvokeModel` permission
- Check if the model is available in your region

**Error: "Invalid model ID"**
- Use inference profile IDs for on-demand access (prefix: `us.anthropic.`)
- Example: `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
