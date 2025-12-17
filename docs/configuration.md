# Configuration

`log-triage` is a Python toolkit that sits between your log collector and an LLM. It reads log files, applies configurable rules to group and classify entries, and emits structured findings and payloads you can forward to a model. The CLI and Web UI share the same YAML configuration so you can run batch jobs, follow live streams, or explore results in a dashboard.

> Tip: You can build and edit configuration (including regexes) directly in the Web UI's config editor and regex lab. It is the easiest way to tweak values safely before saving.

## Core concepts

- **Pipelines:** Reusable processing recipes. Each pipeline defines how to group log lines, how to classify them, which regexes to ignore, and which prompts to use.
- **Modules:** Runtime bindings that attach a pipeline to a specific file path, execution mode (batch/follow), and output options.
- **Findings:** Structured results produced by a classifier. They include severity, reason, counts, context, and optional LLM payloads for downstream tooling.
- **Severity:** One of `WARNING`, `ERROR`, or `CRITICAL`. Severity can be escalated by anomaly detection or manually adjusted in the Web UI.
- **Regex filters:**
  - `ignore_regexes` drop lines before counting.
  - `warning_regexes` and `error_regexes` count matched lines to set severity.
  - The regex lab in the UI helps experiment with these patterns.
- **Addressed & false positives:** Addressed findings track items you reviewed. Marking a finding as a false positive also adds the pattern to the pipeline's ignore list, preventing repeats.

Keep these terms in mind while reading the configuration sections below.

## File structure at a glance

```yaml
pipelines: []
modules: []
llm:
  default_provider: null
  providers: {}
database:
alerts:
  webhook: {}
  mqtt: {}
```

Each top-level key controls a specific area of the system. The example below expands all of them in context.

## Example configuration

```yaml
pipelines:
  - name: homeassistant
    grouping: whole_file
    classifier: regex_counter
    ignore_regexes:
      - "^DEBUG"
    warning_regexes:
      - "warning|timeout"
    error_regexes:
      - "(error|failed)"
    prompt_template: './prompts/homeassistant.txt'

modules:
  - name: homeassistant_follow
    enabled: true
    path: '/var/log/fluent-bit/homeassistant.log'
    mode: 'follow'
    pipeline: 'homeassistant'
    output_format: 'text'
    min_print_severity: 'WARNING'
    stale_after_minutes: 60
    from_beginning: false
    interval: 5
    llm:
      enabled: true
      provider: 'openai'
      emit_llm_payloads_dir: './ha_llm_payloads'
      context_prefix_lines: 2
    alerts:
      webhook:
        enabled: true
        url: 'https://hooks.example.local/logs'

llm:
  default_provider: openai
  providers:
    openai:
      base_url: 'https://api.openai.com'
      model: 'gpt-4o-mini'
      api_key_env: 'OPENAI_API_KEY'

database:
  url: 'sqlite:///./logtriage.db'
  echo: false
 
alerts:
  webhook:
    enabled: false
    url: ''
```

> Tip: The Web UI config editor provides inline hints and validation so you can tweak values safely before saving, and the regex lab lets you test patterns interactively before applying them.

Line-by-line highlights for the example above:

- `pipelines[].grouping`: choose how to split the log stream (`whole_file` or `marker_based`).
- `pipelines[].classifier`: select which classifier implementation to run.
- `ignore_regexes`, `warning_regexes`, `error_regexes`: filter noise or count severity indicators; editable from the Web UI when marking false positives.
- `prompt_template`: path to the prompt file for LLM payloads.
- `modules[].mode`: `batch` scans once; `follow` tails continuously.
- `modules[].from_beginning` and `interval`: follow-mode controls for where to start reading and how often to poll for rotations.
- `modules[].stale_after_minutes`: used by the Web UI to mark **follow** modules as stale when no new lines arrive; batch modules do not use this flag because they run once and exit.
- `modules[].llm`: per-module override for enabling payloads, picking a provider, and setting context lines or output paths.
- `modules[].alerts`: enable webhook or MQTT destinations per module.
- `llm.default_provider` and `llm.providers`: global provider definitions, including where API keys come from.
- `database.url`: connection string for persisting findings so the Web UI can query history.

### LLM sampling controls

- **temperature** tweaks randomness in completions (0–2). Lower values keep replies steady and deterministic; higher values make them more creative or exploratory. See OpenAI's parameter guide for examples: https://platform.openai.com/docs/guides/text-generation/parameter-details.
- **top_p** (nucleus sampling) limits generation to the smallest set of tokens whose cumulative probability stays under the threshold. Smaller values constrain the model to safer choices; larger values behave more like an unrestricted search. Same reference: https://platform.openai.com/docs/guides/text-generation/parameter-details.

## Pipelines

Pipelines describe how to interpret a log stream. You can edit their regexes and templates directly from the Web UI config editor and regex lab.

```yaml
pipelines:
  - name: homeassistant
    grouping:
      type: whole_file  # or marker
    classifier: regex_counter  # or rsnapshot_basic
    ignore_regexes:
      - "^DEBUG"
    warning_regexes:
      - "warning|timeout"
    error_regexes:
      - "(error|failed)"
```

- **grouping** controls how log lines are chunked before classification.
  - `whole_file` treats the entire file as a single chunk, ideal for log files that represent one run or backup job.
  - `marker` looks for start/end markers (for example, rsnapshot headers) and groups lines between markers.
  - Optional keys under `grouping`:
    - `start_regex` / `end_regex`: boundaries for marker grouping.
    - `only_last`: when `true`, process only the final grouped chunk (useful when a batch module should analyze just the most recent run instead of the entire historical log).
- **classifier** picks the rule set used to score each chunk. You can plug in your own; see [Classifiers](classifiers.md) for guidance.
- **ignore_regexes** lets you drop known-noise lines before counting errors and warnings. False positives you mark in the UI are added here automatically.
- **warning_regexes / error_regexes** count matches separately so the classifier can choose the highest severity observed.

When using `marker_based` grouping, ensure your regexes align with the markers your log source emits (for example, rsnapshot's `^\d{4}/\d{2}/\d{2}` headers). Everything between two markers is treated as a single event, so place warnings/errors accordingly.

## Modules

Modules connect pipelines to real log files and decide when to run. You can adjust module paths, modes, and LLM settings directly from the Web UI config editor without touching the YAML manually.

```yaml
modules:
  - name: homeassistant_follow
    enabled: true
    path: '/var/log/fluent-bit/homeassistant.log'
    mode: 'follow'           # 'batch' or 'follow'
    pipeline: 'homeassistant'
    output_format: 'text'    # 'text' or 'json'
    min_print_severity: 'WARNING'
    stale_after_minutes: 60  # used by the Web UI for activity indicators
```

### Follow mode options

Follow mode tails a single file and keeps up with rotations. Combine these options to control responsiveness and resource usage:

- `from_beginning`: when true, reads the whole file before tailing. Set to `false` for a tail-only experience similar to `tail -F`.
- `interval`: poll frequency (in seconds) for new lines and rotation detection. Lower values react faster but hit the file system more often.
- `reload_on_change`: CLI flag that reloads configuration when `config.yaml` changes so follow-mode modules pick up regex, grouping, and prompt updates without restarting.
- `stale_after_minutes`: used only for follow-mode modules to trigger "stale" warnings in the Web UI when no new lines are seen within the window.

### LLM options per module

```yaml
modules:
  - name: homeassistant_follow
    llm:
      enabled: true
      provider: 'openai'
      prompt_template: './prompts/homeassistant.txt'
      emit_llm_payloads_dir: './ha_llm_payloads'
      context_prefix_lines: 2
```

Each module can choose whether to generate payloads. The prompt template is formatted with placeholders such as `{pipeline}`, `{file_path}`, `{severity}`, `{reason}`, `{error_count}`, `{warning_count}`, and `{line_count}`. A prompt fragment might look like this:

```
Pipeline {pipeline} reported {error_count} errors and {warning_count} warnings in {file_path}. Context:
{context}
```

`context` is assembled using the lines around each grouped chunk; tune `context_prefix_lines` to include enough lead-in. Edit templates from the Web UI to experiment quickly without restarting the CLI.

### Alerts

Modules support multiple alert channels:

- **Webhook:**

  ```yaml
  alerts:
    webhook:
      enabled: true
      url: 'https://hooks.example.local/logs'
      headers:
        Authorization: 'Bearer <token>'
  ```

- **MQTT:**

  ```yaml
  alerts:
    mqtt:
      enabled: true
      host: 'mqtt.example.local'
      port: 1883
      topic: 'alerts/logs'
      tls: false
  ```

## LLM providers

Providers are defined globally and referenced by name from modules.

```yaml
llm:
  default_provider: openai
  providers:
    openai:
      base_url: 'https://api.openai.com'
      model: 'gpt-4o-mini'
      api_key_env: 'OPENAI_API_KEY'
```

When only one provider is defined, modules inherit it automatically. The CLI reads the API key from the named environment variable.

## Database

To persist findings for the Web UI, configure a database connection:

```yaml
database:
  url: 'sqlite:///./logtriage.db'
  echo: false
```

SQLite and Postgres URLs are both supported. When omitted, the Web UI stores data in memory and only reflects the current session.

## RAG (Retrieval-Augmented Generation)

RAG enhances log analysis by automatically retrieving relevant documentation from knowledge bases and including it in LLM prompts. This provides more accurate, context-aware responses with proper citations.

### Global RAG Configuration

```yaml
rag:
  enabled: true
  cache_dir: "./rag_cache"
  vector_store:
    persist_directory: "./rag_vector_store"
  embedding:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"  # or "cuda" for GPU acceleration
    batch_size: 32
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
    max_chunks: 10
```

### Module-Level RAG Configuration

Add RAG configuration to individual modules:

```yaml
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

#### RAG Configuration Options

- **cache_dir**: Directory for storing cloned Git repositories
- **vector_store.persist_directory**: Directory for ChromaDB vector storage
- **embedding.model_name**: SentenceTransformer model for embeddings
- **embedding.device**: "cpu" or "cuda" for GPU acceleration
- **embedding.batch_size**: Batch size for embedding generation
- **retrieval.top_k**: Maximum number of chunks to retrieve
- **retrieval.similarity_threshold**: Minimum similarity score (0.0-1.0)
- **retrieval.max_chunks**: Maximum chunks to consider during search
- **knowledge_sources**: List of Git repositories containing documentation
- **include_paths**: Glob patterns for selecting files from repositories

For detailed RAG setup instructions, see the [RAG Quick Start Guide](RAG-QuickStart.md) and [RAG documentation](RAG.md).
