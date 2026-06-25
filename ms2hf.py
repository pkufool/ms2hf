"""Command line tools for syncing repositories between ModelScope and Hugging Face.

Design goals:
- run in storage-constrained environments such as Google Colab and Gitpod;
- avoid cloning a full repository by downloading, uploading, and deleting one file at a time;
- expose both ``ms2hf`` and ``hf2ms`` console entry points from the same module.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

# Both platforms distinguish model and dataset repositories; keep one internal vocabulary.
REPO_TYPE_MODEL = "model"
REPO_TYPE_DATASET = "dataset"
SUPPORTED_REPO_TYPES = (REPO_TYPE_MODEL, REPO_TYPE_DATASET)


@dataclass(frozen=True)
class FileItem:
    """A single file in a source repository.

    ``path`` is the repository-relative path. ``size`` is optional because not
    every platform API returns file sizes.
    """

    path: str
    size: Optional[int] = None


@dataclass(frozen=True)
class FileResult:
    """Result of syncing a single file, used for final statistics and errors."""

    path: str
    status: str  # uploaded | skipped | failed | dry-run
    message: str = ""


def _normalise_repo_type(repo_type: str) -> str:
    """Normalize user-provided repository type aliases to model/dataset."""

    value = (repo_type or "").lower().strip()
    aliases = {
        "model": REPO_TYPE_MODEL,
        "models": REPO_TYPE_MODEL,
        "dataset": REPO_TYPE_DATASET,
        "datasets": REPO_TYPE_DATASET,
        "data": REPO_TYPE_DATASET,
    }
    if value not in aliases:
        raise argparse.ArgumentTypeError(
            f"Unsupported repo type {repo_type!r}; choose from: model, dataset"
        )
    return aliases[value]


def _hf_repo_type(repo_type: str) -> Optional[str]:
    """Convert the internal repo type to the value expected by huggingface_hub.

    Hugging Face uses ``None`` for model repositories and ``"dataset"`` for
    dataset repositories.
    """
    return None if repo_type == REPO_TYPE_MODEL else REPO_TYPE_DATASET


def _split_namespace_name(repo_id: str) -> Tuple[str, str]:
    """Split a ``namespace/name`` repository id into namespace and name.

    Some ModelScope dataset APIs require namespace and name as separate
    arguments instead of a single repo id.
    """

    if "/" not in repo_id:
        raise ValueError(
            f"ModelScope dataset repo id must be in 'namespace/name' form, got {repo_id!r}"
        )
    namespace, name = repo_id.split("/", 1)
    if not namespace or not name:
        raise ValueError(f"Invalid repo id {repo_id!r}; expected 'namespace/name'")
    return namespace, name


def _compile_regexes(patterns: Optional[Sequence[str]]) -> List[re.Pattern[str]]:
    """Pre-compile include/exclude regexes so syntax errors fail early."""

    compiled: List[re.Pattern[str]] = []
    for pattern in patterns or []:
        compiled.append(re.compile(pattern))
    return compiled


def _matches_filters(
    path: str,
    include_patterns: Sequence[re.Pattern[str]],
    exclude_patterns: Sequence[re.Pattern[str]],
) -> bool:
    """Return whether a repository path passes include/exclude filters.

    Include rules are evaluated first. If at least one include regex is set, the
    path must match one of them. Exclude rules then remove any matching path.
    """

    if include_patterns and not any(p.search(path) for p in include_patterns):
        return False
    if exclude_patterns and any(p.search(path) for p in exclude_patterns):
        return False
    return True


def _coerce_size(value: object) -> Optional[int]:
    """Best-effort conversion of SDK Size/size fields to int."""

    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class HuggingFaceBackend:
    """Hugging Face platform adapter.

    The higher-level sync flow only needs list/download/upload/create operations
    and does not depend on huggingface_hub parameter details directly.
    """

    def __init__(self, token: Optional[str] = None):
        from huggingface_hub import HfApi

        self.token = token
        self.api = HfApi(token=token)

    def list_files(self, repo_id: str, repo_type: str, revision: Optional[str]) -> List[FileItem]:
        paths = self.api.list_repo_files(
            repo_id=repo_id,
            repo_type=_hf_repo_type(repo_type),
            revision=revision,
            token=self.token,
        )
        return [FileItem(path=p) for p in paths]

    def download_file(
        self,
        repo_id: str,
        repo_type: str,
        revision: Optional[str],
        file_path: str,
        local_dir: Path,
        force: bool,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type=_hf_repo_type(repo_type),
            revision=revision,
            local_dir=str(local_dir),
            token=self.token,
            force_download=force,
        )
        return Path(downloaded)

    def upload_file(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        local_path: Path,
        commit_message: str,
        target_revision: Optional[str] = None,
    ) -> None:
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=_hf_repo_type(repo_type),
            revision=target_revision,
            token=self.token,
            commit_message=commit_message,
        )

    def create_repo(self, repo_id: str, repo_type: str, private: bool) -> None:
        self.api.create_repo(
            repo_id=repo_id,
            repo_type=_hf_repo_type(repo_type),
            token=self.token,
            private=private,
            exist_ok=True,
        )


class ModelScopeBackend:
    """ModelScope platform adapter.

    ModelScope uses different APIs and parameters for model and dataset listing
    and downloads. This class exposes an interface similar to HuggingFaceBackend.
    """

    def __init__(self, token: Optional[str] = None):
        from modelscope.hub.api import HubApi

        self.token = token
        self.api = HubApi()
        if token:
            self.api.login(token)

    def list_files(self, repo_id: str, repo_type: str, revision: Optional[str]) -> List[FileItem]:
        # ModelScope model repositories can be listed recursively in one call.
        if repo_type == REPO_TYPE_MODEL:
            files = self.api.get_model_files(
                model_id=repo_id,
                revision=revision or "master",
                recursive=True,
                use_cookies=bool(self.token),
            )
            return [
                FileItem(path=f["Path"], size=_coerce_size(f.get("Size") or f.get("size")))
                for f in files
                if f.get("Type") != "tree" and f.get("Path")
            ]

        # ModelScope dataset repository trees are paginated.
        namespace, name = _split_namespace_name(repo_id)
        page_number = 1
        page_size = 1000
        out: List[FileItem] = []
        while True:
            response = self.api.list_repo_tree(
                dataset_name=name,
                namespace=namespace,
                revision=revision or "master",
                root_path="/",
                recursive=True,
                page_number=page_number,
                page_size=page_size,
            )
            data = response.get("Data") or {}
            files = data.get("Files") or []
            out.extend(
                FileItem(path=f["Path"], size=_coerce_size(f.get("Size") or f.get("size")))
                for f in files
                if f.get("Type") != "tree" and f.get("Path")
            )
            if len(files) < page_size:
                break
            page_number += 1
        return out

    def download_file(
        self,
        repo_id: str,
        repo_type: str,
        revision: Optional[str],
        file_path: str,
        local_dir: Path,
        force: bool,
    ) -> Path:
        # ModelScope's download helpers decide whether to reuse cached files.
        # local_files_only=False ensures an online fetch is allowed.  If a file
        # already exists in our per-file temporary directory it is safe to reuse.
        if repo_type == REPO_TYPE_MODEL:
            from modelscope import model_file_download

            downloaded = model_file_download(
                model_id=repo_id,
                file_path=file_path,
                revision=revision or "master",
                local_dir=str(local_dir),
                local_files_only=False,
            )
        else:
            from modelscope import dataset_file_download

            downloaded = dataset_file_download(
                dataset_id=repo_id,
                file_path=file_path,
                revision=revision or "master",
                local_dir=str(local_dir),
                local_files_only=False,
            )
        if not downloaded:
            raise RuntimeError(f"ModelScope did not return a local path for {file_path}")
        return Path(downloaded)

    def upload_file(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        local_path: Path,
        commit_message: str,
        target_revision: Optional[str] = None,
    ) -> None:
        if target_revision:
            # The current ModelScope SDK upload_file API commits to the default branch.
            # We keep the parameter for a symmetric caller API but cannot pass it through.
            pass
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=self.token,
            commit_message=commit_message,
        )

    def create_repo(self, repo_id: str, repo_type: str, private: bool) -> None:
        visibility = 1 if private else 5
        try:
            if repo_type == REPO_TYPE_MODEL:
                self.api.create_model(model_id=repo_id, visibility=visibility)
            else:
                namespace, name = _split_namespace_name(repo_id)
                self.api.create_dataset(dataset_name=name, namespace=namespace, visibility=visibility)
        except Exception as exc:  # noqa: BLE001 - SDK exception classes vary by version.
            text = str(exc).lower()
            if "exist" in text or "already" in text or "重复" in text:
                return
            raise


def _format_size(size: Optional[int]) -> str:
    """Format a byte count into a human-readable string."""

    if size is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


def _make_commit_message(template: str, direction: str, file_path: str) -> str:
    """Render the per-file commit message from the user-provided template."""

    return template.format(direction=direction, path=file_path, file=file_path)


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    """Remove empty parent directories created under the working directory."""

    try:
        current = path.parent
        stop_at = stop_at.resolve()
        while current.resolve() != stop_at and stop_at in current.resolve().parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent
    except FileNotFoundError:
        return


def _safe_cleanup(path: Path, stop_at: Path) -> None:
    """Safely remove a downloaded temporary file.

    Cleanup is limited to temporary directories created for this sync run and
    does not touch unrelated files in the user-provided working directory.
    """

    try:
        if path.exists() or path.is_symlink():
            path.unlink()
            _remove_empty_parents(path, stop_at)
    except IsADirectoryError:
        shutil.rmtree(path, ignore_errors=True)
    except FileNotFoundError:
        pass


def _with_retries(
    action: Callable[[], None],
    *,
    max_retries: int,
    retry_sleep: float,
    retry_label: str,
) -> None:
    """Retry a network download/upload action.

    SDK exception classes differ across platforms and versions, so this helper
    catches broad exceptions, reports them, waits, and retries.
    """

    last_error: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            action()
            return
        except Exception as exc:  # noqa: BLE001 - report and retry SDK/network failures.
            last_error = exc
            if attempt >= max_retries:
                break
            print(f"{retry_label} failed ({attempt}/{max_retries}): {exc}; retrying in {retry_sleep:g}s", file=sys.stderr)
            time.sleep(retry_sleep)
    assert last_error is not None
    raise last_error


def sync_repositories(
    *,
    direction: str,
    source_repo: str,
    target_repo: str,
    repo_type: str,
    source_revision: Optional[str],
    target_revision: Optional[str],
    work_dir: Path,
    include_regex: Optional[Sequence[str]],
    exclude_regex: Optional[Sequence[str]],
    force: bool,
    workers: int,
    max_retries: int,
    retry_sleep: float,
    hf_token: Optional[str],
    ms_token: Optional[str],
    create_target: bool,
    private: bool,
    dry_run: bool,
    keep_local: bool,
    commit_message_template: str,
) -> List[FileResult]:
    """Run one repository sync job.

    Core flow:
    1. choose source and target backends from the direction;
    2. list source files and existing target files;
    3. filter files with include/exclude regexes;
    4. skip existing target paths unless --force is set;
    5. download, upload, and clean up each file.
    """

    from tqdm.auto import tqdm

    # Choose source and target backends from the command direction.
    if direction == "ms2hf":
        source = ModelScopeBackend(token=ms_token)
        target = HuggingFaceBackend(token=hf_token)
        source_name = "ModelScope"
        target_name = "Hugging Face"
    elif direction == "hf2ms":
        source = HuggingFaceBackend(token=hf_token)
        target = ModelScopeBackend(token=ms_token)
        source_name = "Hugging Face"
        target_name = "ModelScope"
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    if create_target:
        print(f"Ensuring target {target_name} {repo_type} repo exists: {target_repo}")
        target.create_repo(target_repo, repo_type, private=private)

    print(f"Listing source files from {source_name}: {source_repo}")
    source_files = source.list_files(source_repo, repo_type, source_revision)

    # Apply regex filters before downloading to avoid unnecessary transfers.
    include_patterns = _compile_regexes(include_regex)
    exclude_patterns = _compile_regexes(exclude_regex)
    selected = [
        item for item in source_files if _matches_filters(item.path, include_patterns, exclude_patterns)
    ]

    print(f"Found {len(source_files)} source file(s); {len(selected)} selected after filters")

    print(f"Listing target files from {target_name}: {target_repo}")
    try:
        target_existing: Set[str] = {
            item.path for item in target.list_files(target_repo, repo_type, target_revision)
        }
    except Exception as exc:  # noqa: BLE001
        if create_target:
            raise
        raise RuntimeError(
            f"Could not list target repo {target_repo!r}. If it does not exist, rerun with --create-target. Original error: {exc}"
        ) from exc

    skipped_existing = 0 if force else sum(1 for item in selected if item.path in target_existing)
    to_transfer = selected if force else [item for item in selected if item.path not in target_existing]

    print(
        f"Target already has {len(target_existing)} file(s); "
        f"{skipped_existing} selected file(s) will be skipped; "
        f"{len(to_transfer)} file(s) will be transferred"
    )

    if dry_run:
        results = [FileResult(item.path, "skipped", "already exists") for item in selected if item.path in target_existing and not force]
        results.extend(FileResult(item.path, "dry-run", _format_size(item.size)) for item in to_transfer)
        return results

    work_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, workers)
    max_retries = max(1, max_retries)

    initial_skips = [
        FileResult(item.path, "skipped", "already exists on target")
        for item in selected
        if item.path in target_existing and not force
    ]

    def transfer_one(item: FileItem) -> FileResult:
        # Use one temporary directory per file. This makes cleanup reliable and
        # prevents concurrent transfers from overwriting each other's files.
        file_work_dir = Path(
            tempfile.mkdtemp(prefix="ms2hf-file-", dir=str(work_dir))
        )
        local_path: Optional[Path] = None
        try:
            def download_action() -> None:
                # Download only the current file into its temporary directory.
                nonlocal local_path
                local_path = source.download_file(
                    source_repo,
                    repo_type,
                    source_revision,
                    item.path,
                    file_work_dir,
                    force,
                )

            _with_retries(
                download_action,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                retry_label=f"Download {item.path}",
            )
            if local_path is None:
                raise RuntimeError("download did not produce a local path")

            commit_message = _make_commit_message(commit_message_template, direction, item.path)

            def upload_action() -> None:
                # Temporary files are cleaned up in finally unless keep_local is set.
                assert local_path is not None
                target.upload_file(
                    target_repo,
                    repo_type,
                    item.path,
                    local_path,
                    commit_message,
                    target_revision,
                )

            _with_retries(
                upload_action,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                retry_label=f"Upload {item.path}",
            )
            return FileResult(item.path, "uploaded", _format_size(item.size))
        except Exception as exc:  # noqa: BLE001
            return FileResult(item.path, "failed", str(exc))
        finally:
            # Eager cleanup keeps the tool usable in low-disk environments.
            if local_path is not None and not keep_local:
                _safe_cleanup(local_path, file_work_dir)
            if not keep_local:
                shutil.rmtree(file_work_dir, ignore_errors=True)

    results: List[FileResult] = list(initial_skips)
    if not to_transfer:
        return results

    if workers == 1:
        # Single-threaded mode is slower but minimizes local disk usage.
        iterator: Iterable[FileItem] = tqdm(to_transfer, desc="Syncing files", unit="file")
        for item in iterator:
            result = transfer_one(item)
            results.append(result)
            if result.status == "failed":
                tqdm.write(f"FAILED {result.path}: {result.message}")
        return results

    # Multi-threaded mode is useful when network bandwidth and disk space allow it.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(transfer_one, item): item for item in to_transfer}
        with tqdm(total=len(future_to_item), desc="Syncing files", unit="file") as progress:
            for future in as_completed(future_to_item):
                result = future.result()
                results.append(result)
                if result.status == "failed":
                    tqdm.write(f"FAILED {result.path}: {result.message}")
                progress.update(1)
    return results


def _env_token(name: str) -> Optional[str]:
    """Read a token from the environment; treat an empty string as unset."""

    value = os.environ.get(name)
    return value if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ms2hf",
        description="Synchronize model or dataset repositories between ModelScope and Hugging Face one file at a time.",
    )
    parser.add_argument(
        "source_repo",
        help="Source repo id, for example 'k2-fsa/OpenDialog'.",
    )
    parser.add_argument(
        "target_repo",
        help="Target repo id, for example 'your-name/OpenDialog'.",
    )
    parser.add_argument(
        "--repo-type",
        "--type",
        default=REPO_TYPE_MODEL,
        type=_normalise_repo_type,
        choices=SUPPORTED_REPO_TYPES,
        help="Repository type: model or dataset. Default: model.",
    )
    parser.add_argument("--source-revision", help="Source branch, tag, or commit. Defaults to the platform default.")
    parser.add_argument("--target-revision", help="Target revision/branch used when supported by the target platform.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Local working directory. Defaults to the current directory.",
    )
    parser.add_argument(
        "--include-regex",
        action="append",
        help="Only sync files whose repo path matches this Python regular expression. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        help="Exclude files/folders whose repo path matches this Python regular expression. Can be repeated.",
    )
    parser.add_argument("--force", action="store_true", help="Upload even if the target already has the same path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to transfer concurrently. Default: 1 to minimize disk usage.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per download/upload step. Default: 3.")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Seconds to wait between retries. Default: 10.")
    parser.add_argument("--hf-token", default=_env_token("HF_TOKEN"), help="Hugging Face token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--ms-token", default=_env_token("MS_TOKEN"), help="ModelScope token. Defaults to MS_TOKEN env var.")
    parser.add_argument(
        "--create-target",
        action="store_true",
        help="Create the target repo first if the platform API supports it.",
    )
    parser.add_argument("--private", action="store_true", help="Create target repo as private when using --create-target.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be synced without downloading or uploading.")
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep downloaded files in the work directory instead of deleting them after upload.",
    )
    parser.add_argument(
        "--commit-message-template",
        default="{direction}: sync {path}",
        help="Per-file commit message template. Available fields: {direction}, {path}, {file}.",
    )
    return parser


def _summarize(results: Sequence[FileResult]) -> int:
    counts = {"uploaded": 0, "skipped": 0, "failed": 0, "dry-run": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print("\nSync result:")
    print(f"  uploaded: {counts.get('uploaded', 0)}")
    print(f"  skipped:  {counts.get('skipped', 0)}")
    print(f"  dry-run:  {counts.get('dry-run', 0)}")
    print(f"  failed:   {counts.get('failed', 0)}")

    failures = [r for r in results if r.status == "failed"]
    if failures:
        print("\nFailed files:", file=sys.stderr)
        for result in failures[:50]:
            print(f"  {result.path}: {result.message}", file=sys.stderr)
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more", file=sys.stderr)
        return 1
    return 0


def run_cli(direction: str, argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    parser.prog = direction
    args = parser.parse_args(argv)

    try:
        results = sync_repositories(
            direction=direction,
            source_repo=args.source_repo,
            target_repo=args.target_repo,
            repo_type=args.repo_type,
            source_revision=args.source_revision,
            target_revision=args.target_revision,
            work_dir=args.work_dir,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            force=args.force,
            workers=args.workers,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            hf_token=args.hf_token,
            ms_token=args.ms_token,
            create_target=args.create_target,
            private=args.private,
            dry_run=args.dry_run,
            keep_local=args.keep_local,
            commit_message_template=args.commit_message_template,
        )
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return _summarize(results)


def main_ms2hf(argv: Optional[Sequence[str]] = None) -> int:
    return run_cli("ms2hf", argv)


def main_hf2ms(argv: Optional[Sequence[str]] = None) -> int:
    return run_cli("hf2ms", argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    executable = Path(sys.argv[0]).name.lower()
    direction = "hf2ms" if executable.endswith("hf2ms") else "ms2hf"
    return run_cli(direction, argv)


if __name__ == "__main__":
    raise SystemExit(main())
