import os
from importlib import metadata

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def get_runtime_requirements(cfg):
    defaults = {
        "albumentations": ">=2.0.0,<3.0.0",
        "opencv-python": ">=4.8.0,<5.0.0",
        "torch": ">=2.2.0,<3.0.0",
        "torchvision": ">=0.17.0,<1.0.0",
    }
    runtime_cfg = cfg.get("runtime", {}) if cfg else {}
    return runtime_cfg.get("required_versions", defaults)


def validate_runtime_versions(required_versions):
    mismatches = []
    missing = []

    for package_name, spec in required_versions.items():
        try:
            installed = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            missing.append(package_name)
            continue

        if spec:
            spec_set = SpecifierSet(spec)
            if Version(installed) not in spec_set:
                mismatches.append((package_name, installed, spec))

    if not missing and not mismatches:
        return

    lines = ["Runtime dependency check failed."]
    if missing:
        lines.append("Missing packages: " + ", ".join(sorted(missing)))
    if mismatches:
        lines.append("Version mismatches:")
        for package_name, installed, spec in mismatches:
            lines.append(f"- {package_name}: installed {installed}, required {spec}")

    lines.append("Reinstall with: pip install -r requirements.txt")
    raise RuntimeError("\n".join(lines))


def ensure_folders(cfg):
    outputs_dir = cfg.get("logging", {}).get("save_dir", "outputs")
    models_dir = cfg.get("logging", {}).get("model_dir", "models")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return outputs_dir, models_dir


def get_data_paths(cfg):
    data_cfg = cfg.get("data", {})
    data_dir = data_cfg.get("data_dir", "dataset_split")
    train_dir = os.path.join(data_dir, data_cfg.get("train_dir", "train"))
    test_dir = os.path.join(data_dir, data_cfg.get("test_dir", "test"))
    return data_dir, train_dir, test_dir
