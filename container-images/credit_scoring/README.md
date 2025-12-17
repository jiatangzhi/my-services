# 🪙 Credit Scoring Microservice with MLP & FastAPI

This project provides a production-ready inference microservice, packaged with Docker. It uses a Multilayer Perceptron (MLP) model trained with PyTorch to evaluate a credit applicant’s risk in real time.

## 🎯 Key Features

- Modern API: Built with FastAPI, offering high performance and automatic interactive documentation (Swagger UI).

- Deep Learning Model: Uses PyTorch for predictions, enabling complex neural network architectures.

- Deployment Ready: Fully dockerized, ensuring a consistent environment and seamless deployment.

- Integrated Preprocessing: The scikit-learn preprocessing pipeline is embedded, guaranteeing that inference data is processed exactly as during training.

## 🏁 Build & Run Guide

### Step 1: Prepare Artifacts
- Make sure the trained model (`.pt`) and the preprocessor (`.joblib`) are available in the folder: `python/credit_scoring/models/`.

### Step 2: Build the Docker Image
- Navigate to the root directory `my_services/` and run the following command to build the image.

```bash
docker build -t my_services/credit-scoring-mlp:1.0 -f container-images/credit_scoring/Dockerfile .
```
### Step 3: Run the Docker Container
- Once the image is built, start the container with:

```bash
 docker run -d -p 8000:8000 --name credit-scoring-service my_services/credit-scoring-mlp:1.0
```

### Step 4: Verify the Service
- Open your browser and go to the following URL to access the interactive API documentation:

```bash
http://localhost:8000/docs
```

## 📝 How to Use the API (Making a Prediction)
The main endpoint is `/mlp_demo`. You can send a POST request with the applicant’s data in JSON format.

- Option A: Using the Interactive Documentation (Swagger)

    - Open http://localhost:8000/docs.

    - Expand the `POST /mlp_demo`.

    - Click "Try it out".

    - Edit the Request Body with the applicant’s data.

    - Click "Execute" 👉 The model response will appear instantly.

- Option B: Using cURL from the Terminal

    - Run the following example request:

        ```bash
        curl -X 'POST' \
        'http://localhost:8000/mlp_demo' \
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


## ⚙️ Container Management
Useful Docker commands to manage the service:

- Stop the container:

    ```bash
    docker stop credit-scoring-service  
    ```

- View real-time logs:
    ```bash
    docker logs -f credit-scoring-service 
    ```

- Restart a stopped container:
    ```bash
    docker start credit-scoring-service 
    ```

- Stop and remove the container completely:
    ```bash
    docker stop credit-scoring-service && docker rm credit-scoring-service
    ```