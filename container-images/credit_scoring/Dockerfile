# --- Stage 1: Build ---
# We use a slim Python base image to reduce the image size.
FROM python:3.11-slim as builder

# Environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Working directory inside the container
WORKDIR /app

# Copy only the requirements file first to take advantage of Docker layer caching
COPY container-images/credit_scoring/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final ---
# Create the final image starting from a clean base
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy the installed dependencies from the build stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the credit_scoring project source code
COPY python/credit_scoring/ .

# Expose the port on which the application will run
EXPOSE 8080

# Command to start the Uvicorn server when the container starts
CMD ["bash","-lc","uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
