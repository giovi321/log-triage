# RAG Service - Standalone Documentation Retrieval

## Overview

The RAG (Retrieval-Augmented Generation) service is a standalone FastAPI application that provides documentation retrieval capabilities to LogTriage. It runs as a separate process to improve performance and responsiveness of the WebUI and CLI components.

## Architecture

### Before (Integrated)
- WebUI initializes RAG locally on startup (slow)
- CLI doesn't use RAG in stream mode
- Heavy embedding models block main processes

### After (Standalone Service)
- RAG service runs independently on port 8091
- WebUI/CLI make lightweight HTTP API calls
- Service can be scaled and restarted independently
- Fallback to local RAG if service unavailable

## Installation

Install with FastAPI dependencies:
```bash
pip install '.[webui]'  # Includes FastAPI and uvicorn
# or
pip install fastapi uvicorn requests
```

## Configuration

Add to your `config.yaml`:

```yaml
rag:
  enabled: true
  service_url: "http://127.0.0.1:8091"  # RAG service URL
  cache_dir: "./rag_cache"
  vector_store:
    persist_directory: "./rag_vector_store"
  embedding:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"  # Use "cuda" for GPU acceleration
    batch_size: 32
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
    max_chunks: 10
```

## Usage

### Starting the RAG Service

```bash
# Using the entry point
logtriage-rag --config ./config.yaml --host 127.0.0.1 --port 8091

# Or directly with Python
python -m logtriage.rag.service --config ./config.yaml
```

### Starting WebUI (with RAG service)

```bash
# Start RAG service first
logtriage-rag --config ./config.yaml &

# Then start WebUI (will automatically connect to RAG service)
logtriage-webui --config ./config.yaml
```

### Starting CLI (with RAG service)

```bash
# Start RAG service first
logtriage-rag --config ./config.yaml &

# Then start CLI (will automatically connect to RAG service)
logtriage --config ./config.yaml --module homeassistant
```

## API Endpoints

### Health Check
```http
GET /health
```
Returns service health status.

### Status
```http
GET /status
```
Returns RAG system status including repository information.

### Retrieve Documentation
```http
POST /retrieve/{module_name}
Content-Type: application/json

{
  "file_path": "/var/log/app.log",
  "pipeline_name": "homeassistant",
  "finding_index": 1,
  "severity": "ERROR",
  "message": "Connection failed",
  "line_start": 100,
  "line_end": 105,
  "excerpt": ["Error line 1", "Error line 2"]
}
```

### Update Knowledge Base
```http
POST /update-knowledge
```
Triggers knowledge base reindexing.

### Update Module Configuration
```http
POST /module/{module_name}/config
Content-Type: application/json

{
  "module_name": "homeassistant",
  "enabled": true,
  "knowledge_sources": [
    {
      "repo_url": "https://github.com/home-assistant/developers.home-assistant",
      "branch": "master",
      "include_paths": ["docs/**/*.md"]
    }
  ]
}
```

## Performance Benefits

1. **Faster WebUI Startup**: WebUI no longer waits for embedding models to load
2. **Better Responsiveness**: RAG operations don't block the main UI thread
3. **Independent Scaling**: RAG service can be restarted without affecting WebUI
4. **Resource Isolation**: Heavy RAG operations run in separate process
5. **Graceful Degradation**: Falls back to local RAG if service unavailable

## Troubleshooting

### Service Won't Start
- Check if FastAPI dependencies are installed
- Verify config file exists and is valid
- Check if port 8091 is available

### WebUI Shows "RAG service unavailable"
- Ensure RAG service is running: `logtriage-rag --config ./config.yaml`
- Check service URL in config matches actual service
- Verify network connectivity between services

### Performance Issues
- Consider using GPU acceleration: `device: "cuda"` in config
- Reduce `batch_size` if memory is limited
- Use smaller embedding model for faster initialization

## Development

### Running with Auto-reload
```bash
logtriage-rag --config ./config.yaml --reload
```

### API Documentation
When the service is running, visit:
- Swagger UI: http://127.0.0.1:8091/docs
- ReDoc: http://127.0.0.1:8091/redoc

## Migration from Local RAG

1. Install FastAPI dependencies
2. Update `config.yaml` with `service_url`
3. Start RAG service: `logtriage-rag --config ./config.yaml`
4. Restart WebUI/CLI applications

The system will automatically detect and use the RAG service if available, with graceful fallback to local RAG if needed.
