# Retrieval-Augmented Generation (RAG) in Log-Triage

This document explains how to configure and use the RAG functionality in log-triage to provide AI-powered log analysis with documentation context.

## Overview

The RAG system enhances log-triage by:

1. **Knowledge Source Management**: Automatically clones and manages Git repositories containing documentation
2. **Document Processing**: Chunks documentation into searchable pieces with metadata
3. **Vector Storage**: Stores document embeddings in ChromaDB for fast semantic search
4. **Contextual Analysis**: Retrieves relevant documentation snippets and injects them into LLM prompts
5. **Citations**: Provides source citations for AI responses

## Configuration

### Global RAG Settings

Add a `rag` section to your existing `config.yaml`:

```yaml
rag:
  enabled: true
  cache_dir: "./rag_cache"
  vector_store_dir: "./rag_vector_store"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda" for GPU acceleration
  batch_size: 32
  top_k: 5
  similarity_threshold: 0.7
  max_chunks: 10
```

### Module-Specific RAG Configuration

Add RAG configuration to individual modules in your existing `config.yaml`:

```yaml
modules:
  my_service:
    path: "/var/log/my_service"
    pipeline: "my_pipeline"
    llm:
      enabled: true
      provider: "openai"
    rag:
      enabled: true
      knowledge_sources:
        - repo_url: "https://github.com/myorg/service-docs"
          branch: "main"
          include_paths:
            - "docs/**/*.md"      # All .md files in docs and subdirectories
            - "README.md"          # Specific file in root
            - "troubleshooting/*.md"  # .md files in troubleshooting directory only
        - repo_url: "https://github.com/myorg/troubleshooting-guide"
          branch: "main"
          include_paths:
            - "**/*.md"           # All .md files in entire repository
            - "**/*.markdown"     # All .markdown files in entire repository
```

#### include_paths

Use glob patterns to specify which files to include:
- `"docs/**/*.md"` - All .md files in docs directory and all subdirectories
- `"README.md"` - Specific file in repository root
- `"troubleshooting/*.md"` - .md files in troubleshooting directory only
- `"**/*.md"` - All .md files in entire repository
- `"docs/**/*.rst"` - All .rst files in docs and subdirectories
- `"source/**/*.markdown"` - All .markdown files in source directory and subdirectories

**Note**: File extensions are specified directly in the glob patterns. No separate include_extensions field is needed.

## Features

### Knowledge Source Management

- **Automatic Cloning**: Repositories are cloned to a local cache directory
- **Incremental Updates**: Only reindexes when commits change
- **Security**: Git hooks are disabled for security
- **File Filtering**: Processes files based on configurable include_paths (glob patterns)

### Document Processing

- **Smart Chunking**: Markdown files are split by headings, plain text by paragraphs
- **Overlap**: Chunks include overlap for better context
- **Metadata**: Each chunk stores source file, heading, and commit information
- **Target Size**: Chunks aim for 200-500 tokens each

### Semantic Search

- **Embeddings**: Uses SentenceTransformer models for semantic understanding
- **Vector Storage**: ChromaDB provides persistent, queryable storage
- **Similarity Filtering**: Results are filtered by similarity threshold
- **Deduplication**: Duplicate content is automatically removed

### LLM Integration

- **Context Injection**: Relevant documentation is added to LLM prompts
- **Citations**: AI responses include source references
- **Fallback**: Gracefully handles cases with no relevant documentation
- **RAG-Enhanced Prompts**: All built-in prompt templates have been updated to leverage RAG context

## Prompt Templates and RAG Integration

The prompt templates in the `prompts/` directory have been enhanced to take full advantage of RAG functionality. When RAG is enabled, the system automatically:

1. **Retrieves Relevant Documentation**: For each finding, the RAG system searches the knowledge base for relevant documentation chunks
2. **Injects Context**: Retrieved documentation is appended to the prompt after the log chunk
3. **Enables Citations**: The LLM is instructed to use reference numbers in brackets to cite sources

### Updated Prompt Templates

All prompt templates now include RAG-aware instructions:

- **ai_opinion_summary.txt**: Enhanced to reference documentation and include citations
- **authentik.txt**: Added documentation grounding and citation instructions for Authentik-specific analysis
- **homeassistant.txt**: Updated to leverage Home Assistant documentation from knowledge base
- **nextcloud.txt**: Enhanced to use Nextcloud documentation for more accurate troubleshooting
- **rsnapshot.txt**: Updated to reference backup documentation and best practices

### How RAG Context is Added

The RAG context is automatically appended to prompts in this format:

```
--- Relevant Documentation ---

1. [Document Heading]
[Document content]

2. [Document Heading]
[Document content]
...
```

Each prompt template now instructs the LLM to:
- Use the provided documentation to ground analysis
- Cite relevant sources using reference numbers in brackets
- Provide more accurate, context-aware responses

### Example: Before vs After RAG

**Before RAG:**
```
You are analyzing authentik logs...
Focus on authentication/outpost configuration problems...
```

**After RAG:**
```
You are analyzing authentik logs... You will be provided with relevant documentation from the knowledge base to help with your analysis.
Focus on authentication/outpost configuration problems... Use the provided documentation to ground your analysis and cite relevant sources using the reference numbers in brackets.
```

This ensures that when RAG is enabled, the LLM actively references the provided documentation and includes proper citations in its responses.

## Dashboard Integration

The web dashboard shows RAG status including:

- **Overall Status**: Whether RAG is enabled and operational
- **Repository Status**: Individual repository health and indexing status
- **Storage Information**: Vector store location and chunk counts
- **Update Status**: Last indexed commits and reindexing needs

## AI Logs Explorer

The AI logs explorer has been updated to:

- **Remove Prompt Builder**: No longer requires manual prompt construction
- **Add LLM Query Button**: Direct LLM analysis for individual findings
- **Display Citations**: Shows source references in AI responses
- **Show Usage Information**: Displays token usage and model information

## Usage Examples

### Basic Setup

1. Configure your `config.yaml` with RAG settings
2. Restart log-triage to initialize the RAG system
3. The dashboard will show repository indexing progress
4. Once indexing completes, findings will include contextual analysis

### Troubleshooting Common Issues

#### No RAG Results
- Check that RAG is enabled in both global and module configuration
- Verify repositories are accessible and contain documentation
- Check the dashboard for indexing errors

#### Performance Issues
- Consider using a smaller embedding model for faster processing
- Adjust `similarity_threshold` to balance quality vs. quantity
- Use GPU acceleration by setting `device: "cuda"` if available

#### Memory Usage
- Limit the number of repositories per module
- Monitor vector store size and clean up if needed

## Advanced Configuration

### Complete Example

Here's a complete example of RAG configuration in your existing `config.yaml`:

```yaml
rag:
  enabled: true
  cache_dir: "./rag_cache"
  vector_store_dir: "./rag_vector_store"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32
  top_k: 5
  similarity_threshold: 0.7
  max_chunks: 10

pipelines:
  web_service:
    match:
      filename_regex: "web.*\.log"
    classifier:
      type: regex_counter
      error_regexes:
        - "ERROR.*Exception"

modules:
  web_api:
    path: "/var/log/web_api"
    pipeline: "web_service"
    llm:
      enabled: true
      provider: "openai"
    rag:
      enabled: true
      knowledge_sources:
        - repo_url: "https://github.com/myorg/web-service-docs"
          branch: "main"
          include_paths:
            - "docs/*.md"
            - "README.md"
```

## API Integration

The RAG system integrates with the existing LLM API:

```bash
# Query LLM for a specific finding with RAG context
curl -X POST "http://localhost:8000/llm/query_finding" \
  -H "Content-Type: multipart/form-data" \
  -F "finding_id=123" \
  -F "provider=openai"
```

Response includes citations:
```json
{
  "provider": "openai",
  "model": "gpt-4",
  "content": "The error indicates a database connection timeout. Check the connection pool settings and ensure the database server is accessible. [1]",
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 45
  },
  "citations": ["[1] Database Connection Issues (database-troubleshooting.md)"]
}
```

## Monitoring and Maintenance

### Log Monitoring

RAG operations are logged with appropriate levels:
- `INFO`: Successful operations, indexing progress
- `WARNING`: Non-critical issues, empty results
- `ERROR`: Failed operations, configuration problems

### Storage Management

- **Cache Directory**: Contains cloned Git repositories
- **Vector Store**: ChromaDB persistence directory
- **Cleanup**: Remove old repositories manually if needed

### Performance Monitoring

Monitor:
- Indexing time for large repositories
- Query response times
- Memory usage during embedding generation
- Vector store size growth

## Security Considerations

- **Git Hooks**: Automatically disabled for security
- **File Types**: Only processes whitelisted file extensions
- **Network Access**: Repositories are downloaded to local cache
- **Access Control**: Uses existing log-triage authentication

## Troubleshooting

### Common Errors

1. **"Failed to clone repository"**
   - Check repository URL and branch
   - Verify network connectivity
   - Check authentication if using private repos

2. **"No embeddings generated"**
   - Verify embedding model is accessible
   - Check available memory/disk space
   - Review model configuration

3. **"Empty search results"**
   - Lower similarity threshold
   - Check document content quality
   - Verify query construction

### Debug Mode

Enable debug logging:
```yaml
logging:
  level: "DEBUG"
  loggers:
    logtriage.rag: "DEBUG"
```

## Migration from Previous Versions

If upgrading from a version without RAG:

1. Add RAG configuration to your existing `config.yaml` (see examples above)
2. Install new dependencies: `pip install -e .`
3. Restart log-triage
4. Monitor dashboard for initial indexing
5. No database migration required

## Future Enhancements

Planned improvements:
- Support for additional vector databases
- Advanced chunking strategies
- Real-time repository monitoring
- Multi-modal document processing
- Custom embedding model support
