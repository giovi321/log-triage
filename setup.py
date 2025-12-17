from pathlib import Path
from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
readme_path = BASE_DIR / "README.md"
version: dict = {}
version_path = BASE_DIR / "logtriage" / "version.py"
exec(version_path.read_text(), version)

setup(
    name="logtriage",
    version=version["__version__"],
    description="Rule-based log triage and LLM payload generator.",
    long_description=readme_path.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="log-triage maintainers",
    url="https://github.com/giovi321/log-triage",
    license="GPL-3.0",
    packages=find_packages(exclude=("tests", "docs", "scripts")),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "GitPython>=3.1.0",
        "markdown>=3.4.0",
    ],
    extras_require={
        "webui": [
            "fastapi>=0.109",
            "uvicorn>=0.23",
            "jinja2>=3.1",
            "python-multipart>=0.0.5",
            "passlib[bcrypt]>=1.7.4,<1.7.5",
            "bcrypt<4.2.0",
            "sqlalchemy>=2.0",
            "itsdangerous>=2.1",
        ],
        "alerts": ["paho-mqtt>=1.6"],
    },
    entry_points={
        "console_scripts": [
            "logtriage=logtriage.cli:main",
            "logtriage-webui=logtriage.webui.__main__:main",
            "logtriage-rag=logtriage.rag.service:main",
        ],
    },
    package_data={
        "logtriage": [
            "webui/templates/*.html",
            "webui/assets/*",
            "webui/context_hints.json",
        ]
    },
    include_package_data=True,
)
