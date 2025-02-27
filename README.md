# LLM Comparative Analysis Tool

This project provides a web application to compare responses from multiple Large Language Models (LLMs) side by side, with automated analysis. It generates a comprehensive HTML report containing the models' answers to a given query, along with similarity metrics and visualizations.

## Author
- Isaac Harlem (isaacharlem@uchicago.edu)

## Features

- **Multi-Model Query:** Ask a question once and get responses from multiple LLMs.
- **Automated Metrics:** Computes cosine similarity between response embeddings to find the most consistent answer, plus BLEU and ROUGE-L scores if a reference answer is provided.
- **Visualizations:** Includes PCA plots of embeddings, similarity distribution histograms, heatmaps, and summary bar charts.
- **Interactive Web UI:** Easy-to-use React frontend with real-time progress updates as models generate responses.
- **Reports Archive:** Each run produces an HTML report saved in a persistent volume (`reports/` directory) for later viewing.

## Architecture

- **Frontend:** React app (SPA) that connects via WebSocket to the backend for live updates. Users input the query and model names here.
- **Backend:** FastAPI server that orchestrates calls to the LLM engine (Ollama) and computes metrics. Exposes a WebSocket endpoint for progress and serves the final reports.
- **LLM Engine:** [Ollama](https://github.com/ollama/ollama) is used to run LLMs and embedding models locally (supports CPU, Apple MPS, and NVIDIA GPUs).
- **Containerization:** Docker Compose is used to containerize the frontend, backend, and Ollama services for easy deployment across different environments (development laptops, servers, Slurm clusters, etc.).

## Repository Structure

After cloning, your repository should have the following structure:

COGS_PROJ/ ├── .gitignore # Ignore node_modules, build artifacts, etc. ├── README.md ├── docker-compose.yml ├── reports/ # Persisted HTML reports (empty initially) ├── backend/ │ ├── Dockerfile.backend │ ├── app.py │ ├── model_comparison.py │ └── requirements.txt ├── docker/ │ ├── Dockerfile.ollama │ └── entrypoint.sh └── frontend/ ├── Dockerfile.frontend ├── package.json ├── package-lock.json # Generated by npm install ├── public/ │ └── index.html └── src/ ├── App.js └── index.js


*Note:* The `node_modules/` folder is not committed (and is listed in `.gitignore`).

## Prerequisites

- **Docker** (or Podman with podman-compose) installed on your system.
- Sufficient system resources to run the chosen LLMs. (For heavy models, a GPU is recommended: Apple M1/M2 or NVIDIA GPU with CUDA.)
- Ensure your system has internet access for downloading Docker images and models.
- (Optional) If using BLEU metrics, note that NLTK data may be downloaded on the first run.

## Setup and Running

Follow these steps after cloning the repository:

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd COGS_PROJ
```

### 2. (Optional) Initialize Git
If not already a Git repository, initialize it:
```bash
git init
```

### 3. Frontend Setup
Navigate to the frontend directory and install the dependencies.
```bash
cd frontend
npm install
```
This will generate a consistent package-lock.json. (The node_modules/ folder is not committed.)

### 4. Ensure the Ollama Entrypoint Script Is Executable
Return to the repository root and set executable permission on the entrypoint script:
```bash
cd ..
chmod +x docker/entrypoint.sh
```

### 5. Build and Start All Containers
From the repository root, run:
```bash
docker-compose up --build
```
This command will:
1. Build the Frontend: Using Dockerfile.frontend to install Node.js dependencies, build the React app, and serve it via Nginx.
2. Build the Backend: Using Dockerfile.backend to install Python dependencies and run the FastAPI app.
3. Build the Ollama Service: Using docker/Dockerfile.ollama with the provided entrypoint.sh script, which pulls the required models (smollm:135m, deepseek-r1:1.5b, qwen:1.8b) and then starts the Ollama server.

### 6. Access the Application
Once the containers are running:
* Frontend:
    Open your browser at http://localhost:8080 (for production build via Nginx) or http://localhost:3000 if using the development server.
* Backend:
    The FastAPI backend (with WebSocket and API endpoints) runs on http://localhost:8000.
* Ollama Service:
    Runs on http://localhost:11434.

### 7. Using the Web Application
On the frontend page:
1. Enter a Query: Type your question or prompt.
2. List Models: Provide model identifiers as a comma-separated list (e.g., deepseek-r1:1.5b, qwen:1.8b). Ensure these models match those pulled by the Ollama service.
3. Set Number of Responses: Choose the number of responses each model should generate.
4. Reference Answer (Optional): Enter a reference answer to compute BLEU and ROUGE-L scores.
5. Click Generate Report.
Real-time progress will be shown via WebSocket updates. Once completed, the final report (an HTML page) will be displayed in an embedded frame and saved in the reports/ directory.

### 8. Stopping the Application
To stop all services, press Ctrl+C in the terminal running Docker Compose or run:
```bash
docker-compose down
```
The generated reports will remain in the reports/ directory.

## Customization and Troubleshooting
* Model Configuration:
    To change the models being pulled, update the entrypoint script in docker/entrypoint.sh. For example, modify the commands to pull different models.
* NLP Metrics & Visualizations:
    The backend (in backend/model_comparison.py) computes cosine similarity, BLEU, and ROUGE-L metrics, and generates various plots. You can extend these metrics by editing that file.
* GPU Support:
    For NVIDIA GPU support, ensure Docker is configured with the NVIDIA Container Toolkit. You can adjust GPU settings in docker-compose.yml if needed. For Apple M1/M2, Docker Desktop should automatically handle hardware acceleration.
* Default 404 on Root:
    The FastAPI backend is API-only and may return 404 for the root path. You can add a simple route in backend/app.py if desired.
* Ollama Service:
    If the Ollama service fails to start, try re-running chmod +x docker/entrypoint.sh and docker-compose up --build.
