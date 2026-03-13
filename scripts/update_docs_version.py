#!/usr/bin/env python3
"""Synchronize MkDocs version with logtriage/version.py."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "logtriage" / "version.py"
MKDOCS_FILE = ROOT / "mkdocs.yml"


def extract_version() -> str:
    """Return the __version__ string from logtriage/version.py."""
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        raise ValueError("Unable to find __version__ in logtriage/version.py")
    return match.group(1)


def update_mkdocs_version(version: str) -> bool:
    """Update the version value under the extra block in mkdocs.yml.

    Returns True if a change was made, False otherwise.
    """
    lines = MKDOCS_FILE.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    in_extra = False
    version_updated = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith("extra:") and indent == 0:
            in_extra = True
        elif in_extra and stripped and indent == 0:
            in_extra = False

        if in_extra and stripped.startswith("version:"):
            new_line = f"  version: {version}"
            if line != new_line:
                version_updated = True
            line = new_line
        new_lines.append(line)

    if not any(l.lstrip().startswith("version:") and (len(l) - len(l.lstrip())) == 2 for l in new_lines if l.strip()):
        # If no version was present under extra, append it.
        insert_index = next((i for i, l in enumerate(new_lines) if l.lstrip().startswith("extra:") and (len(l) - len(l.lstrip())) == 0), None)
        if insert_index is None:
            new_lines.append("extra:")
            insert_index = len(new_lines) - 1
        new_lines.insert(insert_index + 1, f"  version: {version}")
        version_updated = True

    new_content = "\n".join(new_lines) + "\n"
    current_content = MKDOCS_FILE.read_text(encoding="utf-8")
    if new_content != current_content:
        MKDOCS_FILE.write_text(new_content, encoding="utf-8")
        return True
    return version_updated


def main() -> None:
    version = extract_version()
    changed = update_mkdocs_version(version)
    if changed:
        print(f"mkdocs.yml updated to version {version}")
    else:
        print(f"mkdocs.yml already set to version {version}")


if __name__ == "__main__":
    main()
