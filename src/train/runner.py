from typing import Dict


def _normalize_device(device: str) -> str:
    val = str(device).strip().lower()
    if val in {"gpu", "cuda"}:
        return "0"
    return str(device)


def run_training(cfg: Dict) -> None:
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
