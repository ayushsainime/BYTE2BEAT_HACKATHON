# save as: setup_project.py
# run from project root: python setup_project.py

from pathlib import Path

ROOT = Path.cwd()

# dataset source must already exist
dataset_src = ROOT / "dataset_split"
if not dataset_src.exists():
    raise FileNotFoundError("dataset_split folder not found in current directory.")

# beginner-friendly project structure (keeps dataset_split as main data source)
folders = [
    "models",
    "outputs",
    "outputs/plots",
    "outputs/reports",
    "notebooks",
]

files = [
    "app.py",
    "train.py",
    "predict.py",
    "README.md",
    # keep your existing config.yaml and requirements.txt if they already exist
]

for folder in folders:
    path = ROOT / folder
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder}")
    else:
        print(f"Exists folder:  {folder}")

for file in files:
    path = ROOT / file
    if not path.exists():
        path.touch()
        print(f"Created file:   {file}")
    else:
        print(f"Exists file:    {file}")

print("\nDone. Using dataset from:", dataset_src)
