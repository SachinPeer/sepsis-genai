# GenAI Sepsis Early Warning System

AI-powered sepsis prediction and risk stratification system with clinical guardrails.

## Overview

This system provides real-time sepsis risk assessment using a 3-stage AI pipeline:

1. **Pre-Processing**: Normalizes patient vitals and validates data
2. **AI Analysis**: Generates risk scores and clinical rationale
3. **Clinical Guardrail**: Deterministic safety layer for critical thresholds

## Features

- **Risk Scoring**: 0-100 risk score with 6-hour predictive probability
- **Priority Classification**: Critical, Urgent, Elevated, Standard
- **Clinical Rationale**: AI-generated explanation for each assessment
- **Safety Guardrails**: Configurable thresholds that override AI when critical
- **Hot-Reload**: Update guardrail thresholds without restart
- **Batch Processing**: Process multiple patients simultaneously

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Azure OpenAI API access

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/sepsis-genai.git
cd sepsis-genai

# Copy environment template
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Install dependencies
pip install -r requirements.txt

# Run locally
python api.py
```

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Single patient classification |
| `/classify-batch` | POST | Batch patient classification |
| `/health` | GET | Service health check |
| `/guardrail/thresholds` | GET | Current guardrail config |
| `/guardrail/reload` | POST | Hot-reload guardrail config |
| `/docs` | GET | Interactive API documentation |

## Example Request

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{
    "patient_id": "P001",
    "vitals": {
      "HR": 118,
      "SBP": 85,
      "MAP": 58,
      "Temp": 39.2,
      "Resp": 24,
      "O2Sat": 91,
      "WBC": 18.5,
      "Lactate": 3.2
    },
    "notes": "Patient appears confused and lethargic"
  }'
```

## Example Response

```json
{
  "request_id": "req_abc123",
  "patient_id": "P001",
  "status": "success",
  "risk_score": 85,
  "priority": "Critical",
  "sepsis_probability_6h": "High (>70%)",
  "clinical_rationale": "Elevated lactate, hypotension, and tachycardia indicate significant sepsis risk...",
  "alert_level": "CRITICAL",
  "alert_color": "red",
  "action_required": "Immediate evaluation, blood cultures, antibiotics within 1 hour",
  "guardrail_override": true,
  "override_reasons": ["MAP < 65 mmHg (58)", "Lactate ≥ 2.0 mmol/L (3.2)"]
}
```

## Project Structure

```
sepsis-genai/
├── api.py                          # FastAPI application
├── genai_pipeline.py               # 3-stage prediction pipeline
├── genai_inference_service.py      # Amazon Bedrock (Claude) integration
├── guardrail_service.py            # Clinical safety guardrail
├── genai_clinical_guardrail.json   # Configurable thresholds (SME-editable, Docker-bundled)
├── Dockerfile, docker-compose.yml, requirements.txt
│
├── preprocessing/
│   └── genai_preprocess.py         # Data preprocessing / narrative builder
│
├── samples/                        # Example JSON payloads for manual testing
│   ├── genai_test_patients.json
│   ├── test_notes.json
│   └── test_trend.json
│
├── docs/                           # All project documentation
│   ├── architecture/               # How the system works (prompts, PHI, scaling, Mongo design)
│   ├── planning/                   # PROJECT_TRACKER, execution plan, study plan, competitive
│   ├── guides/                     # How-to material (feedback loop, etc.)
│   ├── llm-comparison/             # Multi-LLM bake-off results (+ legacy scripts)
│   ├── generate_pptx.py            # Builds executive overview deck
│   └── generate_storyboard.py      # Builds clinical storyboard deck
│
├── presentations/
│   ├── current/                    # Live decks (DeepDive v6, DayInLife, HospitalConfig, etc.)
│   └── archive/                    # Superseded versions (v1–v5, older leadership decks)
│
├── validation/                     # Validation studies (PhysioNet, eICU-CRD, MIMIC-IV-planned)
│   ├── docs/                       # Reports & plans — start here
│   ├── eicu_cohort_builder.py      # Current pipeline (eICU)
│   ├── run_eicu_validation.py      #
│   ├── select_cohort_v5.py         # PhysioNet locked baseline (R7)
│   ├── results/                    # CSV / JSON run outputs + ROC plots
│   ├── experiments/                # Sensitivity / what-if simulations
│   └── archive/                    # Superseded PhysioNet rounds (R1–R4)
│
├── scripts/                        # Utility scripts (feedback analyzer, etc.)
└── README.md                       # You are here
```

Each top-level folder has its own `README.md` with a deeper map.

## Configuration

### Clinical Guardrail

The `genai_clinical_guardrail.json` file contains all configurable thresholds. Key sections:

- **critical_thresholds**: Hard limits that trigger overrides
- **override_logic**: Rules for septic shock and DIC detection
- **discordance_rules**: Nursing note keywords for silent sepsis
- **history_context_checks**: Patient history considerations

Subject Matter Experts can modify this file to adjust clinical criteria.

### Hot-Reload

After updating `genai_clinical_guardrail.json`:

```bash
curl -X POST "http://localhost:8000/guardrail/reload" \
  -H "x-api-key: your_api_key"
```

## Documentation

- [`docs/README.md`](docs/README.md) — documentation index
- [GenAI Output Explained](docs/architecture/GENAI_OUTPUT_EXPLAINED.md) — what `/classify` returns, field-by-field
- [System Prompt](docs/architecture/prompt.md) — AI reasoning framework (v3.2)
- [PHI & Compliance](docs/architecture/PHI_COMPLIANCE_AND_DATA_HANDLING.md) — HIPAA stance, BAAs, retention
- [Guardrail → Mongo Design](docs/architecture/GUARDRAIL_CONFIG_MONGO_DESIGN.md) — hospital-specific config plan
- [Project Tracker](docs/planning/PROJECT_TRACKER.md) — live task status
- [Validation](validation/README.md) — validation study index (PhysioNet + eICU-CRD + MIMIC-IV plan)

## Security

- API key authentication required for all endpoints
- Never commit `.env` files with credentials
- Use environment variables for all secrets
- Non-root user in Docker container

## License

Proprietary - All rights reserved

## Support

For questions or issues, contact the development team.
