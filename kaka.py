"""Utility script for Hugging Face Space auth + upload workflow.

Usage examples:
  python kaka.py whoami
  python kaka.py login --token hf_xxx
  python kaka.py ensure-space --repo-id ayushsainime/eye_heart_connection_reflex
  python kaka.py upload --repo-id ayushsainime/eye_heart_connection_reflex
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login


DEFAULT_REPO_ID = "ayushsainime/eye_heart_connection_reflex"


def cmd_whoami() -> None:
    api = HfApi()
    info = api.whoami()
    name = info.get("name") if isinstance(info, dict) else None
    print(f"Logged in as: {name or info}")


def cmd_login(token: str | None) -> None:
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Provide --token or set HF_TOKEN environment variable.")
    login(token=token)
    print("Login successful.")


def cmd_ensure_space(repo_id: str) -> None:
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"Space ready: {repo_id}")


def cmd_upload(repo_id: str, token: str | None) -> None:
    api = HfApi(token=token or os.getenv("HF_TOKEN"))

    allow_patterns = [
        "Dockerfile",
        "start_hf_reflex.sh",
        "README.md",
        "pyproject.toml",
        "rxconfig.py",
        ".gitignore",
        ".gitattributes",
        ".dockerignore",
        ".hfignore",
        "configs/**",
        "artifacts/**",
        "api/**",
        "inference/**",
        "models/**",
        "reflex_app/**",
        "utils/**",
        "assets/**",
        "photos/**",
    ]

    ignore_patterns = [
        ".git/**",
        ".venv/**",
        "venv/**",
        "**/__pycache__/**",
        "*.pyc",
        "uploaded_files/**",
        "preprocessed_images/**",
        "experiments/**",
        "data/**",
        "notebooks/**",
        "tests/**",
        "training/**",
        "evaluation/**",
        "datasets/**",
        "visual_output/**",
        "pics_for_frontend_reflex/**",
        ".agents/**",
        ".states/**",
        ".web/**",
    ]

    api.upload_folder(
        folder_path=str(Path(".").resolve()),
        repo_id=repo_id,
        repo_type="space",
        path_in_repo=".",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message="Deploy Reflex Space via upload_folder",
    )
    print(f"Upload complete: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HF Space helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami")

    login_p = sub.add_parser("login")
    login_p.add_argument("--token", default=None)

    ensure_p = sub.add_parser("ensure-space")
    ensure_p.add_argument("--repo-id", default=DEFAULT_REPO_ID)

    upload_p = sub.add_parser("upload")
    upload_p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    upload_p.add_argument("--token", default=None)

    args = parser.parse_args()

    if args.cmd == "whoami":
        cmd_whoami()
    elif args.cmd == "login":
        cmd_login(args.token)
    elif args.cmd == "ensure-space":
        cmd_ensure_space(args.repo_id)
    elif args.cmd == "upload":
        cmd_upload(args.repo_id, args.token)


if __name__ == "__main__":
    main()
