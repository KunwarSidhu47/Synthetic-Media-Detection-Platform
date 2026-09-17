# AWS Cloud Deployment Architecture Guide

This document outlines the production cloud deployment strategy for the **Synthetic Media Detection & Analysis Platform** on **Amazon Web Services (AWS)**.

---

## 🏗️ Target AWS Cloud Architecture

```text
                               ┌────────────────────────────────┐
                               │       AWS Route 53 (DNS)       │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   AWS ALB (Load Balancer)      │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
           ┌────────────────────────┐                      ┌────────────────────────┐
           │   Amazon EC2 Instance  │                      │   Amazon S3 Bucket     │
           │   (Docker Containers)  │                      │ (Raw Video Storage)    │
           │  - FastAPI + PyTorch   │                      └────────────────────────┘
           │  - Nginx + React       │
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │ Amazon RDS PostgreSQL  │
           │  (Session Persistence) │
           └────────────────────────┘
```

---

## 🚀 Step-by-Step Deployment Guide

### 1. Provision EC2 Compute Instance
- **Instance Type**: `g4dn.xlarge` (NVIDIA T4 Tensor Core GPU for accelerated PyTorch ViT inference) or `c6i.2xlarge` (CPU optimized).
- **AMI**: Ubuntu Server 22.04 LTS.
- **Security Group Rules**:
  - Inbound HTTP (Port 80) $\rightarrow$ `0.0.0.0/0`
  - Inbound HTTPS (Port 443) $\rightarrow$ `0.0.0.0/0`
  - Inbound SSH (Port 22) $\rightarrow$ Restricted IP

### 2. Configure EC2 Server Environment
```bash
# SSH into EC2 instance
ssh -i "key.pem" ubuntu@<ec2-public-ip>

# Update package lists and install Docker & Docker Compose
sudo apt update && sudo apt install -y docker.io docker-compose git ffmpeg
sudo usermod -aG docker ubuntu
newgrp docker

# Clone project repository
git clone https://github.com/your-username/deepfake-platform.git
cd deepfake-platform
```

### 3. Provision Amazon RDS (PostgreSQL Database)
- **Database Engine**: PostgreSQL 15.
- **DB Instance Class**: `db.t4g.micro` or `db.m6g.large`.
- **Environment Variable Configuration**:
  ```bash
  export DATABASE_URL="postgresql://user:password@rds-endpoint.amazonaws.com:5432/deepfakedb"
  ```

### 4. Deploy Application Stack via Docker Compose
```bash
# Build and launch background services
docker-compose up -d --build

# Verify container status
docker-compose ps
docker-compose logs -f
```

### 5. SSL/TLS Certificate Setup (Certbot)
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d deepfake-analysis.yourdomain.com
```

---

## 📊 Monitoring & Logging
- **Amazon CloudWatch**: Collect Docker container stdout/stderr logs and metric alarms for CPU/GPU utilization.
- **Health Check Endpoint**: Target `/api/health` for Application Load Balancer target group health checks.
