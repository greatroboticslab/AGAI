# 🚀 Plant Diagnostic System - Deployment Guide

This guide provides comprehensive instructions for deploying the Plant Diagnostic System in various environments.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space minimum

### Software Dependencies

- CUDA 11.8+ (for GPU acceleration)
- cuDNN 8.0+
- PyTorch 2.0+
- Docker (for containerized deployment)

## Local Deployment

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/plant-diagnostic-system.git
cd plant-diagnostic-system

# Create conda environment
conda env create -f environment.yml
conda activate plant-diagnostic

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Model Setup

```bash
# Download required models
mkdir -p checkpoints
mkdir -p llama_weights
mkdir -p plant_diagnostic/models

# Download LLaMA-2-7B weights (place in llama_weights/)
# Download ResNet checkpoint (place in plant_diagnostic/models/)
# Download MiniGPT-v2 checkpoint (place in checkpoints/)
```

### 3. Configuration

```bash
# Copy and edit configuration files
cp eval_configs/minigptv2_eval.yaml.example eval_configs/minigptv2_eval.yaml
# Edit paths in the configuration file
```

### 4. Launch Application

```bash
# Basic launch
python demo_v5.py --cfg-path eval_configs/minigptv2_eval.yaml

# With ResNet anchor for faster inference
python demo_v5.py --cfg-path eval_configs/minigptv2_eval.yaml --resnet-anchor

# Using launch script
./launch_demo_v5.sh --share
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints llama_weights plant_diagnostic/models

# Expose port
EXPOSE 7860

# Set default command
CMD ["python", "demo_v5.py", "--cfg-path", "eval_configs/minigptv2_eval.yaml"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  plant-diagnostic:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./llama_weights:/app/llama_weights
      - ./plant_diagnostic/models:/app/plant_diagnostic/models
      - ./output:/app/output
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### 3. Build and Deploy

```bash
# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f plant-diagnostic
```

## Cloud Deployment

### AWS EC2 Deployment

#### 1. Launch EC2 Instance

- **Instance Type**: g4dn.xlarge or larger (GPU-enabled)
- **AMI**: Deep Learning AMI (Ubuntu 20.04)
- **Storage**: 50GB EBS volume
- **Security Group**: Allow HTTP (80), HTTPS (443), and custom port (7860)

#### 2. Configure Instance

```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 3. Deploy Application

```bash
# Clone repository
git clone https://github.com/your-username/plant-diagnostic-system.git
cd plant-diagnostic-system

# Deploy with Docker Compose
docker-compose up -d
```

### Google Cloud Platform

#### 1. Create VM Instance

```bash
# Create instance with GPU
gcloud compute instances create plant-diagnostic \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=tf-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

#### 2. Deploy Application

```bash
# Connect to instance
gcloud compute ssh plant-diagnostic --zone=us-central1-a

# Follow Docker deployment steps above
```

### Azure Container Instances

#### 1. Create Container Registry

```bash
# Create resource group
az group create --name plant-diagnostic-rg --location eastus

# Create container registry
az acr create --resource-group plant-diagnostic-rg --name plantdiagnostic --sku Basic
```

#### 2. Build and Push Image

```bash
# Build and push to registry
az acr build --registry plantdiagnostic --image plant-diagnostic:latest .
```

#### 3. Deploy Container

```bash
# Deploy container instance
az container create \
    --resource-group plant-diagnostic-rg \
    --name plant-diagnostic \
    --image plantdiagnostic.azurecr.io/plant-diagnostic:latest \
    --cpu 4 \
    --memory 8 \
    --ports 7860 \
    --environment-variables CUDA_VISIBLE_DEVICES=0
```

## Production Configuration

### 1. Environment Variables

```bash
# Create .env file
cat > .env << EOF
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_PATH=/app/checkpoints/checkpoint_best.pth
RESNET_PATH=/app/plant_diagnostic/models/resnet_straw_final.pth
LLAMA_PATH=/app/llama_weights/Llama-2-7b-chat-hf
LOG_LEVEL=INFO
MAX_WORKERS=4
BATCH_SIZE=1
CACHE_SIZE=1000
EOF
```

### 2. Nginx Configuration

```nginx
# /etc/nginx/sites-available/plant-diagnostic
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. SSL Configuration

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 4. Systemd Service

```ini
# /etc/systemd/system/plant-diagnostic.service
[Unit]
Description=Plant Diagnostic System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/plant-diagnostic-system
Environment=PATH=/home/ubuntu/miniconda3/envs/plant-diagnostic/bin
ExecStart=/home/ubuntu/miniconda3/envs/plant-diagnostic/bin/python demo_v5.py --cfg-path eval_configs/minigptv2_eval.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable plant-diagnostic
sudo systemctl start plant-diagnostic
sudo systemctl status plant-diagnostic
```

## Monitoring & Maintenance

### 1. Health Checks

```python
# health_check.py
import requests
import time

def check_health():
    try:
        response = requests.get("http://localhost:7860/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    if check_health():
        print("✅ Service is healthy")
    else:
        print("❌ Service is unhealthy")
```

### 2. Logging Configuration

```python
# logging_config.py
import logging
import logging.handlers

def setup_logging():
    # Create logger
    logger = logging.getLogger('plant_diagnostic')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/plant_diagnostic.log', maxBytes=10*1024*1024, backupCount=5
    )
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 3. Performance Monitoring

```python
# monitor.py
import psutil
import torch
import time

def monitor_system():
    """Monitor system performance."""
    stats = {
        'timestamp': time.time(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }
    
    if torch.cuda.is_available():
        stats.update({
            'gpu_memory_allocated': torch.cuda.memory_allocated(),
            'gpu_memory_reserved': torch.cuda.memory_reserved(),
            'gpu_utilization': torch.cuda.utilization()
        })
    
    return stats
```

### 4. Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Create backup directory
BACKUP_DIR="/backups/plant-diagnostic/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup models
cp -r checkpoints "$BACKUP_DIR/"
cp -r plant_diagnostic/models "$BACKUP_DIR/"

# Backup configuration
cp -r eval_configs "$BACKUP_DIR/"
cp -r train_configs "$BACKUP_DIR/"

# Backup logs
cp -r logs "$BACKUP_DIR/"

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Model Loading Errors

```bash
# Check model files exist
ls -la checkpoints/
ls -la plant_diagnostic/models/
ls -la llama_weights/

# Verify file permissions
chmod 644 checkpoints/*.pth
chmod 644 plant_diagnostic/models/*.pth
```

#### 3. Port Already in Use

```bash
# Find process using port
sudo lsof -i :7860

# Kill process
sudo kill -9 <PID>
```

#### 4. Permission Denied

```bash
# Fix file permissions
sudo chown -R $USER:$USER /path/to/plant-diagnostic-system
chmod +x launch_demo_v5.sh
```

### Performance Optimization

#### 1. GPU Optimization

```python
# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

#### 2. Memory Optimization

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
```

#### 3. Model Optimization

```python
# Compile model for faster inference
model = torch.compile(model, mode="reduce-overhead")
```

### Support

For additional support:

- **Documentation**: [Project Wiki](https://github.com/your-username/plant-diagnostic-system/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/plant-diagnostic-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/plant-diagnostic-system/discussions)

---

**Note**: This deployment guide is regularly updated. Please check for the latest version before deploying.

