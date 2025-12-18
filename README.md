# Credit Scoring ML System: Training, Evaluation & Inference
**End-to-end Credit Scoring Machine Learning system**, designed with **MLOps and production readiness** in mind.
 
 🔹 **Problem**: Predict credit risk (`good` / `bad`) from financial and demographic data  
 🔹 **Model**: PyTorch Multilayer Perceptron (MLP)   
 🔹 **Preprocessing**: Scikit-learn pipeline (training = inference consistency)  
 🔹 **Tracking**: MLflow for experiments, metrics, and artifacts  
 🔹 **Serving**: FastAPI REST API  
 🔹 **Deployment**: Docker (multi-stage) → Cloud Run–ready  

This project demonstrates how to go from **model training → evaluation → versioning → deployment** in a clean, reproducible, production-grade ML system.

---

## What This Project Demonstrates
- Training and evaluating a neural network for a real business problem  
- Tracking experiments and metrics with MLflow  
- Clean separation between training and inference  
- Serving ML models in production via an API  
- Packaging models and code for cloud deployment  

---

## Key Metrics Tracked
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

All metrics are logged to **MLflow** for experiment comparison and reproducibility.

### Metric Visualization & Experiment Analysis
- Metrics, parameters, and artifacts are stored under `my_services/python/credit_scoring/mlruns/`, and enable visual comparison of runs, metric trends, and architectural trade-offs.
- Experiments can be visualized interactively using:
```bash
mlflow ui
```
- Model architecture choices (layers, dropout, batch norm) and training summaries are documented in its subdirectory under `/reports/`.

---

## High-Level Architecture
```text
                   ┌────────────────────┐
                   │   Raw Credit Data  │
                   │ (CSV / structured) │
                   └─────────┬──────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │  Preprocessing Pipeline │
                 │  (scikit-learn)         │
                 └───────────┬─────────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │  PyTorch MLP Model │
                   │  (Training)        │
                   └─────────┬──────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ Model Weights │   │ Preprocessor    │   │ Metrics & Params │
│   (.pt)       │   │ (.joblib)       │   │   (MLflow)       │
└───────────────┘   └─────────────────┘   └──────────────────┘
        │                    │
        └────────────┬───────┘
                     ▼
          ┌──────────────────────────┐
          │   Inference Layer        │
          │  (FastAPI + Predictor)   │
          └────────────┬─────────────┘
                       ▼
              ┌─────────────────┐
              │ REST API Output │
              │ {probability,   │
              │  good / bad}    │
              └─────────────────┘
```

## Deployment Flow (Simplified)
```text
Code + Model Artifacts
        │
        ▼
   Docker Build
   (multi-stage)
        │
        ▼
  Container Image
        │
        ▼
 Cloud Run / Serverless
        │
        ▼
  Real-time Predictions
```

## Why This Is Relevant for Industry

This project reflects real ML engineering workflows, not just notebooks:

✔ Reproducible experiments

✔ Versioned models & preprocessors

✔ Clear separation of concerns

✔ Production inference patterns

✔ Cloud-ready architecture