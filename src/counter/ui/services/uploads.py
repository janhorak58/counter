from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile

import yaml


GT_REQUIRED_KEYS = {"video", "line_name", "in_count", "out_count"}


@dataclass
class UploadResult:
    path: Path
    kind: str
    replaced: bool
    message: str


@dataclass
class AssetItem:
    path: Path
    kind: str
    size_bytes: int
    modified_at: datetime



def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "file"



def ensure_within_root(path: Path, root: Path) -> Path:
    root_res = root.resolve()
    path_res = path.resolve()
    if not str(path_res).startswith(str(root_res)):
        raise ValueError(f"Path escapes allowed root: {path}")
    return path_res



def save_bytes_file(data: bytes, dest_path: Path, *, overwrite: bool = True) -> UploadResult:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    replaced = dest_path.exists()
    if replaced and not overwrite:
        raise FileExistsError(f"File already exists: {dest_path}")

    with dest_path.open("wb") as f:
        f.write(data)

    return UploadResult(
        path=dest_path,
        kind="file",
        replaced=replaced,
        message="replaced" if replaced else "created",
    )



def validate_gt_json_bytes(data: bytes) -> Tuple[bool, str]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return False, f"Invalid JSON: {exc}"

    if not isinstance(obj, dict):
        return False, "GT JSON root must be an object."

    missing = sorted(GT_REQUIRED_KEYS - set(obj.keys()))
    if missing:
        return False, f"Missing required keys: {', '.join(missing)}"

    return True, "valid"



def extract_gt_zip(data: bytes, dest_dir: Path, *, overwrite: bool = True) -> List[UploadResult]:
    out: List[UploadResult] = []
    dest_res = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(BytesIO(data)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            member_name = member.filename.replace("\\", "/")
            if member_name.startswith("/") or ".." in Path(member_name).parts:
                raise ValueError(f"Unsafe zip entry: {member.filename}")

            target = ensure_within_root(dest_res / member_name, dest_res)
            file_bytes = zf.read(member)

            if target.suffix.lower() == ".json":
                ok, msg = validate_gt_json_bytes(file_bytes)
                if not ok:
                    raise ValueError(f"Invalid GT file in zip ({member.filename}): {msg}")

            res = save_bytes_file(file_bytes, target, overwrite=overwrite)
            out.append(UploadResult(path=res.path, kind="gt", replaced=res.replaced, message=res.message))

    return out



def save_video_upload(data: bytes, filename: str, videos_dir: Path, *, overwrite: bool = True) -> UploadResult:
    safe = _safe_name(filename)
    dest = videos_dir / safe
    res = save_bytes_file(data, dest, overwrite=overwrite)
    return UploadResult(path=res.path, kind="video", replaced=res.replaced, message=res.message)



def save_gt_upload(data: bytes, filename: str, gt_dir: Path, *, overwrite: bool = True) -> List[UploadResult]:
    safe = _safe_name(filename)
    suffix = Path(safe).suffix.lower()

    if suffix == ".zip":
        return extract_gt_zip(data, gt_dir, overwrite=overwrite)

    if suffix != ".json":
        raise ValueError("Ground-truth upload must be .json or .zip")

    ok, msg = validate_gt_json_bytes(data)
    if not ok:
        raise ValueError(msg)

    dest = gt_dir / safe
    res = save_bytes_file(data, dest, overwrite=overwrite)
    return [UploadResult(path=res.path, kind="gt", replaced=res.replaced, message=res.message)]



def save_model_upload(
    data: bytes,
    filename: str,
    models_root: Path,
    *,
    backend: str,
    variant: str,
    model_id: str,
    overwrite: bool = True,
) -> UploadResult:
    safe_file = _safe_name(filename)
    model_slug = _safe_name(model_id)
    rel_dir = Path(backend) / variant / model_slug
    dest = models_root / rel_dir / safe_file
    res = save_bytes_file(data, dest, overwrite=overwrite)
    return UploadResult(path=res.path, kind="model", replaced=res.replaced, message=res.message)



def register_model_in_registry(
    *,
    models_yaml_path: Path,
    project_root: Path,
    model_id: str,
    backend: str,
    variant: str,
    weights_path: Path,
    mapping: Optional[Dict[str, int]] = None,
    rfdetr_size: Optional[str] = None,
) -> None:
    if models_yaml_path.exists():
        with models_yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if not isinstance(data, dict):
        data = {}

    if "models" not in data:
        data = {"models": data}

    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError("models.yaml has invalid structure: 'models' must be a mapping")

    rel_weights = weights_path.resolve().relative_to(project_root.resolve())

    entry: Dict[str, Any] = {
        "backend": str(backend),
        "variant": str(variant),
        "weights": str(rel_weights).replace(os.sep, "/"),
    }

    if mapping:
        entry["mapping"] = mapping

    if rfdetr_size and str(rfdetr_size).strip():
        entry["rfdetr_size"] = str(rfdetr_size)

    models[str(model_id)] = entry

    models_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with models_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)



def list_assets(videos_dir: Path, gt_dir: Path, models_root: Path) -> List[AssetItem]:
    out: List[AssetItem] = []

    def _collect(root: Path, kind: str, exts: Optional[Iterable[str]] = None) -> None:
        if not root.exists():
            return

        ext_set = {e.lower() for e in exts} if exts is not None else None
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if ext_set is not None and p.suffix.lower() not in ext_set:
                continue
            st = p.stat()
            out.append(
                AssetItem(
                    path=p,
                    kind=kind,
                    size_bytes=int(st.st_size),
                    modified_at=datetime.fromtimestamp(st.st_mtime),
                )
            )

    _collect(videos_dir, "video", exts={".mp4", ".avi", ".mov", ".mkv"})
    _collect(gt_dir, "gt", exts={".json"})
    _collect(models_root, "model", exts={".pt", ".pth", ".onnx", ".engine", ".bin"})

    out.sort(key=lambda x: x.modified_at, reverse=True)
    return out



def delete_asset(path: Path, *, allowed_roots: Iterable[Path]) -> None:
    p = path.resolve()
    allowed = [r.resolve() for r in allowed_roots]

    if not any(str(p).startswith(str(r)) for r in allowed):
        raise ValueError(f"Delete blocked for path outside allowed roots: {p}")

    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))

    p.unlink()
