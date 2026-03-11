import os
import yaml 


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


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
