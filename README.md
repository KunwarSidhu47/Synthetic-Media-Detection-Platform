# Synthetic Media Detection & Analysis Platform

Production-style AI/ML platform for detecting synthetic and deepfake video content using multimodal signal analysis and interpretable AI explanations.

---

## 🎯 Project Overview

This platform analyzes uploaded videos through multiple complementary detection techniques to deliver interpretable, evidence-backed synthetic media analysis through an interactive web dashboard.

### Key Capabilities
- **Spatial Analysis (ViT)**: Detects frame-level visual artifacts, blending boundaries, and facial unnaturalness using Vision Transformers (`ViT-B/16`).
- **Frequency-Domain Analysis (2D FFT)**: Identifies high-frequency spectral anomalies, GAN checkerboard patterns, and diffusion generation signatures.
- **Temporal Analysis (Bi-LSTM)**: Captures inter-frame inconsistencies and temporal flicker across frame sequences ($N \times 768$ ViT feature vector sequences).
- **Multi-Signal Score Fusion**: Combines spatial, frequency, and temporal confidence scores into a unified detection assessment ($S_{\text{final}} = w_s S_s + w_f S_f + w_t S_t$).
- **LLM Explanation Layer**: Synthesizes model output metadata into clear, human-readable forensic reports with Responsible AI guardrails.
- **Interactive Dashboard**: Built with React, Vite, Recharts, and Vanilla CSS glassmorphic aesthetics.
- **Containerized & CI/CD Ready**: Docker, Docker Compose, Nginx, PostgreSQL compatibility, and GitHub Actions workflow.

---

## 🏗️ Multi-Signal Detection Architecture

```text
                 INPUT VIDEO
                      │
                      ▼
              Frame Extraction
                  (OpenCV)
                      │
                      ▼
             Face Detection & Crop
                  (MediaPipe)
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Spatial     Frequency    Temporal
       Analysis    Analysis     Analysis
          │           │           │
          ▼           ▼           ▼
      ViT-B/16     2D FFT       Bi-LSTM
          │           │           │
          └───────────┼───────────┘
                      ▼
             Multi-Signal Score Fusion
                      │
                      ▼
           Real / Synthetic Prediction
                      │
                      ▼
           FastAPI REST Backend
                      │
                      ▼
         React Glassmorphic Dashboard
                      │
                      ▼
         LLM Natural Language Explanation
```

---

## 📂 Repository Structure

```text
deepfake/
├── data/              # Datasets (raw, processed, SQLite/PostgreSQL db)
├── models/            # ViT-B/16 spatial classifier & Bi-LSTM temporal models
├── notebooks/         # Exploratory data analysis & model training experimentation
├── backend/           # FastAPI services, API routes, database models, LLM layer
│   ├── main.py
│   ├── database.py
│   ├── routes/        # REST endpoints (health, upload, analyze, results, explain)
│   ├── services/      # VideoProcessor, FaceDetector, ViT, FFT, Bi-LSTM, Fusion, LLM
│   └── schemas/       # Pydantic data models
├── frontend/          # React + Vite dashboard and Recharts visualization components
├── scripts/           # Data preprocessing and synthetic video generator scripts
├── tests/             # Unit and integration test suites (32 test cases)
├── docs/              # AWS cloud deployment architecture guide
├── .github/           # GitHub Actions CI/CD pipeline
├── Dockerfile         # Backend container definition
├── docker-compose.yml # Container orchestration
├── requirements.txt   # Python dependency specifications
└── README.md          # Master project specification & documentation
```

---

## 🗺️ 16-Phase Development Roadmap

- [x] **Phase 1**: Project Structure & Environment Setup
- [x] **Phase 2**: Dataset Preparation & Preprocessing Pipeline
- [x] **Phase 3**: Video Processing & MediaPipe Face Detection
- [x] **Phase 4**: Vision Transformer (ViT) Spatial Analysis Model
- [x] **Phase 5**: FFT Frequency-Domain Analysis Engine
- [x] **Phase 6**: LSTM Temporal Analysis Model
- [x] **Phase 7**: Multi-Signal Score Fusion Engine
- [x] **Phase 8**: FastAPI Backend Service & REST Routes
- [x] **Phase 9**: React + Vite Forensic Dashboard
- [x] **Phase 10**: LLM Explanation Layer & Prompt Engineering
- [x] **Phase 11**: Explainability Visualizations & Responsible AI Controls
- [x] **Phase 12**: Database Session Persistence (SQLite / PostgreSQL)
- [x] **Phase 13**: Docker Containerization & Compose Orchestration
- [x] **Phase 14**: Automated Test Suite (32 Passing PyTest Cases)
- [x] **Phase 15**: GitHub Actions CI/CD Pipeline
- [x] **Phase 16**: Cloud Deployment Architecture (AWS EC2 + S3 + RDS)

---

## 🔗 REST API Reference Table

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Returns service health status and PyTorch compute device info |
| `POST` | `/api/upload` | Upload video file (`.mp4`, `.mov`, `.webm`, `.avi`) and get `file_id` |
| `POST` | `/api/analyze` | Execute full multi-signal detection pipeline on uploaded video |
| `GET` | `/api/results/{id}` | Retrieve complete detection result JSON by `analysis_id` |
| `GET` | `/api/results` | List recent analysis session summaries |
| `POST` | `/api/explain` | Generate LLM natural language explanation for analysis result |
| `POST` | `/api/report` | Export structured Markdown forensic analysis report |

---

## 🚀 Getting Started

### 1. Local Python Environment
```bash
# Clone repository
git clone https://github.com/your-username/deepfake-platform.git
cd deepfake-platform

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run backend API server
uvicorn backend.main:app --port 8000 --reload
```

### 2. Local React Dashboard
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000 in your browser
```

### 3. Docker Compose (Full Stack)
```bash
docker-compose up --build
```

### 4. Running Test Suite
```bash
pytest tests/
# Output: 32 passed in 4.25s
```

---

## 💼 Resume Accomplishment Bullets

**Synthetic Media Detection & Analysis Platform | Python, PyTorch, ViT, Bi-LSTM, FastAPI, React**
- Developed a multimodal deepfake detection pipeline combining **Vision Transformer (ViT-B/16) spatial analysis, 2D FFT frequency-domain spectral analysis, and Bi-LSTM temporal sequence modeling**.
- Engineered **FastAPI REST inference APIs** and an interactive React dashboard with Recharts for confidence visualization, suspicious-frame timeline inspection, and markdown report generation.
- Integrated an **LLM explanation layer with structured prompt guardrails** to convert quantitative model metadata into human-interpretable forensic explanations with responsible-AI safeguards.
- Containerized and automated the application using **Docker, Docker Compose, GitHub Actions CI/CD, and AWS cloud deployment architecture (EC2 + S3 + RDS)**.
