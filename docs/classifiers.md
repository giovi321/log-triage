# Classifiers

Classifiers convert grouped log chunks into findings. Built-in options include the regex counter and an rsnapshot-specific heuristic, but you can add your own for project-specific formats.

> Configuration and regexes are easiest to edit in the Web UI. Use the config editor to point pipelines at your new classifier and the regex lab to validate patterns.

## Built-in classifiers

- `regex_counter`: counts warning/error regex matches per line and emits findings with the matching snippet.
- `rsnapshot_basic`: groups rsnapshot runs and applies rsnapshot-aware heuristics.

Select either via `pipelines[].classifier` in `config.yaml`.

## Creating a new classifier

1. **Add a classifier module.** Create a new file under `logtriage/classifiers/` (for example, `custom_http.py`) and implement a function that matches the signature used by the dispatcher:
   - Inputs: `PipelineConfig`, `file_path`, `pipeline_name`, `lines`, `start_line`, `excerpt_limit`, `context_prefix_lines`.
   - Output: a `List[Finding]` populated with severity, message, line numbers, and excerpt context.
2. **Register the classifier.** Update `logtriage/classifiers/__init__.py` to route a new `classifier_type` string to your function, similar to how `rsnapshot_basic` is registered.
3. **Expose configuration.** In `config.yaml`, set `pipelines[].classifier` to your new `classifier_type` (for example, `custom_http`). Add any regexes or options your classifier consumes.
4. **Test with the CLI.** Run `python -m logtriage.cli run --module <module>` and use `--reload-on-change` to iterate quickly while editing code and regexes via the Web UI.

## Authoring tips

- Reuse `Finding` and `Severity` from `logtriage.models` to stay consistent with the Web UI and alerts.
- Use `context_prefix_lines` to include enough preamble for LLM prompts and dashboard excerpts.
- Keep error messages concise; they appear in the CLI, alerts, and UI.
- Add comments in your classifier for any expected log markers so future contributors can align grouping strategies.
