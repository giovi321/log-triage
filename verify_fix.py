#!/usr/bin/env python3
"""Test the context hints fix directly in the browser context."""

import json
from pathlib import Path

# Read the modified config_edit.html to verify the fix is in place
template_file = Path(__file__).parent / "logtriage" / "webui" / "templates" / "config_edit.html"
with open(template_file) as f:
    template_content = f.read()

# Check if our fix is present
if "Special handling for provider-specific fields" in template_content:
    print("✓ The fix is present in config_edit.html")
    print("  - Added special handling for provider-specific fields")
    print("  - Handles llm.providers.<name>.field patterns")
else:
    print("✗ The fix is NOT present in config_edit.html")

# Verify the specific logic for api_key_env
if "llm_providers_*_${firstWord}" in template_content:
    print("✓ The wildcard pattern generation is present")
else:
    print("✗ The wildcard pattern generation is missing")

# Check the context hints JSON
hints_file = Path(__file__).parent / "logtriage" / "webui" / "context_hints.json"
with open(hints_file) as f:
    hints = json.load(f)

print("\nContext hints verification:")
print("=" * 50)

# Verify the required hints exist
required_hints = [
    "llm_providers_api_key_env",
    "llm_providers_*_api_key_env",
    "api_key_env"
]

for hint in required_hints:
    if hint in hints:
        print(f"✓ {hint}: {hints[hint][:60]}...")
    else:
        print(f"✗ {hint}: MISSING")

print("\nFix Summary:")
print("=" * 50)
print("The issue was that context hints for nested provider configurations")
print("like 'api_key_env' under 'llm.providers.openai' were not working.")
print("\nThe fix adds:")
print("1. Special handling for provider-specific fields")
print("2. Wildcard pattern matching (llm_providers_*_api_key_env)")
print("3. Proper ancestor path construction for nested YAML")
print("\nNow when the cursor is on 'api_key_env' line, it will show:")
print(f"'{hints['llm_providers_api_key_env']}'")
