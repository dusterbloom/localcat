import os
from pathlib import Path

# Check which YAML file ASI1 is trying to load
yaml_files = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.yml'))
print("Available YAML files:")
for f in yaml_files:
    print(f"  - {f.name} ({f.stat().st_size} bytes)")

# Check ASI1 default
print("\nASI1 default YAML: ULTRAGROK_V8.2.1_SPACY.yaml")
print("Does it exist?", Path("ULTRAGROK_V8.2.1_SPACY.yaml").exists())

