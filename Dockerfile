# --- Stage 1: Builder ---
# Use a slim Python base imagen to reduce the size.
FROM python:3.11-slim as builder

# Environment variables to optimize
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy the requirements file from its original location 
COPY requirements.txt .

# Install dependencies 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final ---
# Create the final image from a clean base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy the installed dependencies from the build stage  
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the source code. In developpement, this overwrite with a volume.
COPY . .

# Expose the service port
EXPOSE 8000

# Command to run the server with automatic reload for developpement 
CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]