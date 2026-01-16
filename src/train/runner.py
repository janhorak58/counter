from typing import Dict

from src.models.rfdetr import train_rfdetr


def _normalize_device(device: str) -> str:
    val = str(device).strip().lower()
    if val in {"gpu", "cuda"}:
        return "0"
    return str(device)


def run_training(cfg: Dict) -> None:
    model_type = str(cfg.get("model_type", "yolo")).lower()
    if model_type == "rfdetr":
        train_rfdetr(
            model_path=cfg["model"],
            data_yaml=cfg["data_yaml"],
            epochs=int(cfg["epochs"]),
            imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]),
            workers=int(cfg["workers"]),
            device=_normalize_device(cfg["device"]),
            project=cfg["project"],
            name=cfg["name"],
        )
        return

    if model_type != "yolo":
        raise ValueError(f"Unsupported model_type: {model_type}")

    from ultralytics import YOLO

    device = _normalize_device(cfg["device"])
    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data_yaml"],
        epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        workers=int(cfg["workers"]),
        project=cfg["project"],
        name=cfg["name"],
        plots=bool(cfg["plots"]),
        save=bool(cfg["save"]),
        cos_lr=bool(cfg["cos_lr"]),
        device=device,
    )
