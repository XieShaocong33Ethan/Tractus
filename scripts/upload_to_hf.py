#!/usr/bin/env python3
"""
Automated Hugging Face Release Upload Utility.
Uploads Model Cards, weights, and evaluation benchmark datasets to Hugging Face Hub.

Copyright (c) 2026 Ethan Xie (Shaocong Xie). All rights reserved.
Licensed under the Modified MIT License (see LICENSE).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    ".git/**",
    ".env*",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Upload Tractus assets to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID (e.g. Ethan33X/Tractus-0.8B)")
    parser.add_argument(
        "--repo-type",
        type=str,
        choices=["model", "dataset", "space"],
        default="model",
        help="Repository type on Hugging Face",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Local folder to upload (e.g. benchmarks/decision_index_0_3)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Single file to upload (e.g. MODEL_CARD.md)",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="",
        help="Destination path within the HF repository",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="release: upload Tractus official release assets",
        help="Git commit message for HF Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repo if it does not exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the upload without making changes",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== Hugging Face Release Upload Utility ===")
    print(f"Target Repo: {args.repo_id} ({args.repo_type})")
    print(f"Dry Run: {args.dry_run}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        if args.folder:
            print(f"  upload_folder(repo_id='{args.repo_id}', folder_path='{args.folder}', path_in_repo='{args.path_in_repo}')")
        elif args.file:
            print(f"  upload_file(repo_id='{args.repo_id}', path_or_fileobj='{args.file}', path_in_repo='{args.path_in_repo or Path(args.file).name}')")
        print("\nDry run completed successfully.")
        return

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Verify authentication
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info.get('name')}")
    except Exception as e:
        print(f"Authentication warning: {e}")
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
            print("Logged in via HF_TOKEN environment variable.")
        else:
            print("Notice: No HF_TOKEN detected. Ensure you are logged in via 'huggingface-cli login'.")

    # Create repo if not exists
    try:
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
        print(f"Ensured repository exists: {args.repo_id}")
    except Exception as e:
        print(f"Repository check notice: {e}")

    # Perform upload
    if args.folder:
        folder_path = Path(args.folder).resolve()
        if not folder_path.exists():
            print(f"Error: folder {folder_path} does not exist.")
            sys.exit(1)
        print(f"Uploading folder {folder_path} -> {args.repo_id} (path: '{args.path_in_repo}')...")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(folder_path),
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
            ignore_patterns=DEFAULT_IGNORE_PATTERNS,
        )
        print("Folder upload complete!")

    elif args.file:
        file_path = Path(args.file).resolve()
        if not file_path.exists():
            print(f"Error: file {file_path} does not exist.")
            sys.exit(1)
        dest_name = args.path_in_repo or file_path.name
        print(f"Uploading file {file_path} -> {args.repo_id}/{dest_name}...")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_or_fileobj=str(file_path),
            path_in_repo=dest_name,
            commit_message=args.commit_message,
        )
        print("File upload complete!")
    else:
        print("Error: Specify either --folder or --file to upload.")
        sys.exit(1)


if __name__ == "__main__":
    main()
