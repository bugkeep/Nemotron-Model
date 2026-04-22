from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path

import paramiko


DEFAULT_EXCLUDES = [
    ".git",
    ".git/*",
    ".pytest_cache",
    ".pytest_cache/*",
    "__pycache__",
    "__pycache__/*",
    "*/__pycache__/*",
    ".venv",
    ".venv/*",
    ".venv_tinker",
    ".venv_tinker/*",
    "data",
    "data/*",
    "outputs",
    "outputs/*",
    "runs",
    "runs/*",
    "artifacts",
    "artifacts/*",
    "*.jsonl",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.zip",
]


def should_skip(path: Path, root: Path, patterns: list[str]) -> bool:
    relative = path.relative_to(root).as_posix()
    for pattern in patterns:
        if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts: list[str] = []
    current = remote_dir
    while current not in ("", "/"):
        parts.append(current)
        current = os.path.dirname(current)
    for directory in reversed(parts):
        try:
            sftp.stat(directory)
        except FileNotFoundError:
            sftp.mkdir(directory)


def upload_tree(
    sftp: paramiko.SFTPClient,
    local_root: Path,
    remote_root: str,
    excludes: list[str],
) -> None:
    ensure_remote_dir(sftp, remote_root)
    for path in sorted(local_root.rglob("*")):
        if should_skip(path, local_root, excludes):
            continue
        remote_path = f"{remote_root}/{path.relative_to(local_root).as_posix()}"
        if path.is_dir():
            ensure_remote_dir(sftp, remote_path)
            continue
        ensure_remote_dir(sftp, os.path.dirname(remote_path))
        sftp.put(str(path), remote_path)
        print(f"uploaded {path.relative_to(local_root).as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload the local repo to the HFUT SSH machine via SFTP.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default=os.environ.get("HPC_PASSWORD", ""))
    parser.add_argument("--local-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    if not args.password:
        raise SystemExit("Missing password. Pass --password or set HPC_PASSWORD.")

    local_dir = Path(args.local_dir).expanduser().resolve()
    excludes = DEFAULT_EXCLUDES + list(args.exclude)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    try:
        with client.open_sftp() as sftp:
            upload_tree(sftp, local_dir, args.remote_dir, excludes)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
