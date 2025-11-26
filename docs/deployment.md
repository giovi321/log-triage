# Deployment

This site is built with [MkDocs](https://www.mkdocs.org/) and published to GitHub Pages. The repository includes a workflow that installs MkDocs, builds the site, and deploys it to the `gh-pages` branch.

## Local preview

Install dependencies and build the documentation locally:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

MkDocs serves the site on `http://127.0.0.1:8000/` by default and reloads when you edit files in `docs/` or `mkdocs.yml`.

## GitHub Pages workflow

The workflow at `.github/workflows/gh-pages.yml` runs on every push to `main`:

1. Check out the repository.
2. Set up Python and install MkDocs dependencies from `docs/requirements.txt`.
3. Build the site.
4. Deploy the generated HTML to the `gh-pages` branch using the built-in `GITHUB_TOKEN`.

GitHub Pages should be configured to serve from the `gh-pages` branch (root). The workflow runs `mkdocs gh-deploy --force --clean`, which overwrites previous deployments and removes stale files.

### Redeploying

If you need to redeploy the docs without pushing a new commit, make sure you trigger the workflow from the `main` branch (the deployment job only runs on `main`):

1. Go to **Actions → Deploy docs** in the GitHub UI.
2. Click **Run workflow** (available because of the `workflow_dispatch` trigger).
3. Wait for the job to complete; it will rebuild the site and push the fresh output to `gh-pages`.

To redeploy from your workstation instead of GitHub Actions, install the requirements, ensure you have push access, and run:

```bash
python -m pip install -r docs/requirements.txt
mkdocs gh-deploy --force --clean --verbose
```
