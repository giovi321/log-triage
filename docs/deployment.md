# Deployment

This site is built with [MkDocs](https://www.mkdocs.org/) and published to GitHub Pages. The repository includes a workflow that installs MkDocs, builds the site, and deploys it to the `gh-pages` branch.

## Local preview

Install dependencies and build the documentation locally:

```bash
python -m pip install mkdocs mkdocs-material
mkdocs serve
```

MkDocs serves the site on `http://127.0.0.1:8000/` by default and reloads when you edit files in `docs/` or `mkdocs.yml`.

## GitHub Pages workflow

The workflow at `.github/workflows/gh-pages.yml` runs on every push to `main`:

1. Check out the repository.
2. Set up Python and install MkDocs dependencies.
3. Build the site.
4. Deploy the generated HTML to the `gh-pages` branch using the built-in `GITHUB_TOKEN`.

GitHub Pages should be configured to serve from the `gh-pages` branch (root). The workflow runs `mkdocs gh-deploy --force --clean`, which overwrites previous deployments and removes stale files.
