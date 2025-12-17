# RAG Quick Start Guide

This guide helps you get started with RAG (Retrieval-Augmented Generation) in log-triage quickly.

## Prerequisites

- log-triage installed with RAG dependencies
- Git repositories containing documentation
- LLM provider configured (OpenAI, local model, etc.)

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Install RAG dependencies
pip install sentence-transformers chromadb GitPython markdown

# Or install everything with RAG extras
pip install -e .[rag]
```

### 2. Basic Configuration

Add this minimal RAG configuration to your `config.yaml`:

```yaml
rag:
  enabled: true
  cache_dir: "./rag_cache"
  embedding:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"

modules:
  my_service:
    path: "/var/log/my_service"
    pipeline: "my_pipeline"
    llm:
      enabled: true
      provider_name: "openai"
    rag:
      enabled: true
      knowledge_sources:
        - repo_url: "https://github.com/myorg/docs"
          branch: "main"
```

### 3. Start log-triage

```bash
python -m logtriage.webui
```

### 4. Check Dashboard

- Navigate to `http://localhost:8000`
- Check the "Knowledge Base (RAG) Status" section
- Wait for initial indexing to complete

### 5. Test RAG Analysis

- Go to AI Logs Explorer
- Find a log entry marked as a finding
- Click "Query LLM" button
- Review the AI response with citations

## Common Use Cases

### Service Documentation

```yaml
modules:
  api_service:
    rag:
      enabled: true
      knowledge_sources:
        - repo_url: "https://github.com/myorg/api-docs"
          include_paths:
            - "docs/*.md"
            - "troubleshooting/*.md"
```

### Multiple Knowledge Sources

```yaml
modules:
  complex_app:
    rag:
      enabled: true
      knowledge_sources:
        - repo_url: "https://github.com/myorg/user-guide"
          include_paths: ["user/*.md"]
        - repo_url: "https://github.com/myorg/dev-docs"
          include_paths: ["dev/*.md", "api/*.md"]
        - repo_url: "https://github.com/myorg/runbooks"
          include_paths: ["runbooks/*.md"]
```

### Private Repositories

For private repositories, ensure SSH keys or credentials are configured:

```bash
# Set up SSH keys for git access
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

## Performance Tips

### For Better Performance

```yaml
rag:
  embedding:
    device: "cuda"  # If GPU available
    batch_size: 64
  retrieval:
    similarity_threshold: 0.8  # Higher threshold = faster
```

### For Better Quality

```yaml
rag:
  embedding:
    model_name: "sentence-transformers/all-mpnet-base-v2"  # Better model
  retrieval:
    top_k: 10  # More context
    similarity_threshold: 0.6  # Lower threshold = more results
```

## Troubleshooting

### No RAG Results?

1. Check dashboard for RAG status
2. Verify repositories are accessible
3. Check log files for errors
4. Ensure documentation contains relevant content

### Slow Performance?

1. Reduce `batch_size` if memory limited
2. Use smaller embedding model
3. Increase `similarity_threshold`
4. Consider GPU acceleration

### Memory Issues?

1. Reduce `batch_size` to 16 or 8
2. Use CPU instead of GPU if limited VRAM
3. Limit number of knowledge sources
4. Monitor vector store size

## Next Steps

- Review full [RAG documentation](RAG.md)
- Check [example configuration](../config.example.rag.yaml)
- Explore advanced configuration options
- Set up monitoring and alerts

## Support

- Check logs: `tail -f logtriage.log`
- Enable debug logging: Set `logtriage.rag` to DEBUG
- Review dashboard status indicators
- Check repository access and permissions
