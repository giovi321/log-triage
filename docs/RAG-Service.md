# RAG Service - Resilient Documentation Retrieval

## Overview

The RAG (Retrieval-Augmented Generation) service is a standalone FastAPI application that provides documentation retrieval capabilities to LogTriage. It runs as a separate process to improve performance and responsiveness of the WebUI and CLI components.

## Architecture

### Before (Integrated)
- WebUI initializes RAG locally on startup (slow)
- CLI doesn't use RAG in stream mode
- Heavy embedding models block main processes

### After (Standalone & Resilient)
- RAG service runs independently on port 8091
- **Non-blocking startup**: Service starts immediately, initializes in background
- **Graceful degradation**: WebUI/CLI work without RAG if service is down
- **Background updates**: Knowledge base updates don't block API requests
- **Retry logic**: Clients handle temporary failures automatically

## Resilience Features

### 1. Fast Startup
- RAG service starts immediately (doesn't wait for embeddings)
- Heavy initialization runs in background threads
- Health endpoint responds instantly

### 2. Graceful Degradation
- If RAG service is unavailable → WebUI/CLI work without RAG
- No blocking local fallback that hangs startup
- Clear status indicators in WebUI

### 3. Background Operations
- Knowledge base updates run in background
- API requests return empty results during updates (not errors)
- No service downtime during repository updates

### 4. Client Resilience
- Automatic retry with exponential backoff
- Short timeouts (10s) for better responsiveness
- Distinguishes between "service down" and "service initializing"

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

The service will:
1. Start immediately and begin accepting requests
2. Initialize RAG components in background
3. Show "initializing" status during setup
4. Become fully ready once knowledge base is loaded

### Starting WebUI (with RAG service)

```bash
# Start RAG service first
logtriage-rag --config ./config.yaml &

# Then start WebUI (will work even if RAG is still initializing)
logtriage-webui --config ./config.yaml
```

### Starting CLI (with RAG service)

```bash
# Start RAG service first
logtriage-rag --config ./config.yaml &

# Then start CLI (will work even if RAG is still initializing)
logtriage --config ./config.yaml --module homeassistant
```

## API Endpoints

### Health Check
```http
GET /health
```
Returns service health and initialization status:
```json
{
  "status": "healthy",
  "rag_enabled": true,
  "initialization": {
    "started": true,
    "completed": false,
    "updating": true,
    "error": null
  }
}
```

### Status
```http
GET /status
```
Returns RAG system status including repository information. During initialization, returns basic status without blocking.

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
During initialization, returns empty results instead of blocking.

### Update Knowledge Base
```http
POST /update-knowledge
```
Triggers knowledge base reindexing in background.

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

## Behavior During Different States

### Normal Operation
- All RAG features work normally
- WebUI shows full RAG status
- CLI enriches findings with documentation

### During Initialization
- Service responds to health checks immediately
- Retrieval requests return empty results (no errors)
- WebUI shows "initializing" status
- CLI works without RAG enrichment

### During Knowledge Base Updates
- API requests continue working
- Retrieval may use slightly stale data
- Updates run in background
- No service interruption

### When Service is Down
- WebUI starts immediately without RAG
- CLI works without RAG enrichment
- Clear status indicators
- No blocking or hanging

## Performance Benefits

1. **Instant WebUI startup** - No waiting for embedding models
2. **Better responsiveness** - RAG operations don't block UI
3. **Independent scaling** - RAG service can be restarted separately
4. **Resource isolation** - Heavy operations in separate process
5. **Zero downtime** - Updates don't affect other services
6. **Graceful degradation** - System works without RAG

## Troubleshooting

### Service Won't Start
- Check if FastAPI dependencies are installed
- Verify config file exists and is valid
- Check if port 8091 is available

### WebUI Shows "RAG service unavailable"
- Ensure RAG service is running: `logtriage-rag --config ./config.yaml`
- Check service URL in config matches actual service
- Verify network connectivity between services
- Service may still be initializing - check health endpoint

### Performance Issues
- Consider using GPU acceleration: `device: "cuda"` in config
- Reduce `batch_size` if memory is limited
- Use smaller embedding model for faster initialization

### RAG Not Working But Service is Running
- Check if initialization completed: `GET /health`
- Verify knowledge base was loaded successfully
- Check module configurations are correct

## Development

### Running with Auto-reload
```bash
logtriage-rag --config ./config.yaml --reload
```

### API Documentation
When the service is running, visit:
- Swagger UI: http://127.0.0.1:8091/docs
- ReDoc: http://127.0.0.1:8091/redoc

### Monitoring Status
```bash
# Check health and initialization status
curl http://127.0.0.1:8091/health

# Check detailed RAG status
curl http://127.0.0.1:8091/status
```

## Migration from Local RAG

1. Install FastAPI dependencies
2. Update `config.yaml` with `service_url`
3. Start RAG service: `logtriage-rag --config ./config.yaml`
4. Restart WebUI/CLI applications

The system will automatically detect and use the RAG service if available, with graceful fallback to no RAG if needed. **No blocking local fallback** - WebUI will start immediately even if RAG service is down.
