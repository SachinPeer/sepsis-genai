# Scalability & Throughput — Sepsis GenAI System

**TL;DR:** ~10 sec per patient does NOT mean only 30 patients per 5-min cycle. The system processes patients in parallel, not sequentially.

---

## The Concern

| Fact | Value |
|------|-------|
| Average response time per patient | ~10 seconds |
| Hospital polling interval | Every 5 minutes (300 sec) |
| Naive serial calculation | 300 ÷ 10 = **30 patients** |

**But this assumes sequential processing — which is not how the system works.**

---

## Why It Scales — 3 Layers of Parallelism

```
Layer 1: FastAPI (async)     → Multiple requests handled concurrently per pod
Layer 2: EKS (horizontal)    → Multiple pods behind load balancer
Layer 3: Bedrock (cloud)     → 50-200+ concurrent model invocations
```

**Real formula:**

```
Patients per cycle = Concurrent Workers × (300 sec ÷ 10 sec)
```

---

## Capacity by Deployment Size

| Hospital Size | Patients | EKS Pods | Concurrent Calls | Capacity per 5 min | LLM Cost/day |
|--------------|----------|----------|-------------------|-------------------|-------------|
| Small (50 beds) | ~30 | 1 | 5 | 150 | ~$12 |
| Medium (200 beds) | ~100 | 2 | 10 | 300 | ~$40 |
| Large (500 beds) | ~250 | 3–4 | 15–20 | 450–600 | ~$100 |
| Multi-hospital | ~1,000 | 10 | 50 | 1,500 | ~$400 |

*LLM cost at $0.0127/call (Sonnet 4.5), assuming 1 call per patient per 5-min cycle.*

---

## How a 5-Minute Cycle Actually Works

```
T+0s     Red Rover sends batch of patient updates
T+1s     Backend receives, authenticates, fans out to AI
T+2s     ALL patients sent to Bedrock concurrently (not one-by-one)
T+12s    All results back (~10s inference + 2s overhead)
T+13s    Dashboard updates for all patients simultaneously
T+300s   Next cycle begins
```

**Total wall-clock time for 100 patients: ~12 seconds, not 1,000 seconds.**

---

## If We Hit Limits

| Bottleneck | Solution |
|-----------|---------|
| Bedrock concurrent invocation limit | AWS support ticket to increase quota |
| Single pod CPU/memory | Add more EKS pods (horizontal scaling) |
| Massive scale (1,000+ patients) | SQS queue + controlled parallel batches |
| Cost at scale | Switch to Haiku 4.5 for low-risk patients, Sonnet 4.5 for flagged ones |

---

## Already Built

- `/classify-batch` endpoint accepts multiple patients in one API call
- FastAPI async workers handle concurrent Bedrock calls
- EKS + ALB supports horizontal pod autoscaling
- Model swap via env var — can use cheaper model for routine screens

---

*Document created: April 8, 2026 | Model: Claude Sonnet 4.5 | Infra: AWS EKS*
