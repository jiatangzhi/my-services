# Credit Scoring Inference API

This microservice provides a **RESTful API** to predict a credit applicant's risk using a **Multilayer Perceptron (MLP)** neural network built with **PyTorch**.

---

## Overview

The service receives demographic and financial data from an applicant, processes it through a **scikit-learn transformation pipeline**, and passes it to a trained **MLP model** for **binary classification**.

### Key Features

- **High-Performance API**: Built with **FastAPI** for fast inference and low latency.

- **Strong Type Validation**: Uses **Pydantic** to enforce input schemas, including enum validation for fields such as gender, housing, and loan purpose.

- **Deep Learning Inference**: MLP (Multi-Layer Perceptron) model optimized with hidden layers, **Dropout**, and **Batch Normalization**.

- **Modular Architecture**: Clear separation between **server logic** and **inference logic**.

- **Container-Ready**: Fully dockerized and optimized for deployment on **Google Cloud Run**.

- **Integrated Preprocessing**: The scikit-learn preprocessing pipeline is embedded, guaranteeing that inference data is processed exactly as during training.

---

## Build & Run Guide

### Step 1: Prepare Artifacts

- Ensure the trained model (`.pt`) and the preprocessor (`.joblib`) are located in the directory `python/credit_scoring/models/`.

### Step 2: Build the Docker Image

- Navigate to the project root directory `my_services/` and run:

```bash
docker build -t my_services/credit-scoring-mlp:1.0 -f container-images/credit_scoring/Dockerfile .
```

### Step 3: Run the Docker Container

- Once the image is built, start the container with:

```bash
docker run -d -p 8080:8080 --name credit-scoring-service my_services/credit-scoring-mlp:1.0
```

### Step 4: Verify the Service

- Open your browser and navigate to the interactive API documentation:

```bash
http://localhost:8080/docs
```

---

## Using the API (Making a Prediction)

The main endpoint is `/mlp_demo`. You can send a POST request with the applicant's data in JSON format.

- Option A: Using Interactive Documentation (Swagger UI)

  - Open: http://localhost:8080/docs.

  - Expand the POST /mlp_demo endpoint.

  - Click "Try it out".

  - Modify the Request Body with the applicant's data.

  - Click "Execute" to see the model response instantly.

- Option B: Using cURL from the Terminal

  - Run the following example request:

    ```bash
    curl -X 'POST' \
    'http://localhost:8080/mlp_demo' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "Age": 35,
    "Sex": "male",
    "Job": 2,
    "Housing": "own",
    "Saving accounts": "little",
    "Checking account": "moderate",
    "Credit amount": 2500,
    "Duration": 24,
    "Purpose": "car"
    }'
    ```

- Expected Successful Response (200 OK)

  If everything works correctly, you will receive a response like this:

  ```bash
  {
  "prediction": "good",
  "probability": 0.7852
  }
  ```

---

## Container Management

Useful Docker commands to manage the container:

- Stop the container:

  ```bash
  docker stop credit-scoring-service
  ```

- View logs in real time:

  ```bash
  docker logs -f credit-scoring-service
  ```

- Restart a stopped container:

  ```bash
  docker start credit-scoring-service
  ```

- Stop and remove the container:
  ```bash
  docker stop credit-scoring-service && docker rm credit-scoring-service
  ```

---

## Deployment on Google Cloud Platform (GCP)

This service is designed for a serverless architecture using Google Cloud Run.
The CI/CD pipeline is managed via Cloud Build.

### Deployment Flow (Cloud Build)

The file `ops/cloudbuild-credit_scoring_service.yaml` automates the following steps:

1. **Build:** Builds the Docker image using an optimized multi-stage `Dockerfile`.
2. **Push:** Pushes the image to `Artifact Registry` (europe-southwest1-docker.pkg.dev/...).
3. **Deploy:** Deploys the new image to Cloud Run as a managed service.

### Deployment Configuration

- **Region:** `europe-southwest1`
- **Authentication:** `--allow-unauthenticated`
- **Memory:** `1Gi`
- **Port:** `8080`
