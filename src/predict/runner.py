import os
from typing import Dict, Optional

import cv2

from src.core.Counter import Counter
from src.core.DetectedObject import DetectedObject
from src.utils.save_results import save_results_to_csv


def _resolve_model_name(model_path: str, mode: str) -> str:
    if mode == "pretrained":
        return os.path.splitext(os.path.basename(model_path))[0]
    try:
        return os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    except (IndexError, TypeError):
        return os.path.splitext(os.path.basename(model_path))[0]


def run_prediction(cfg: Dict) -> Optional[Dict[str, Dict]]:
    paths = cfg.get("paths", {})
    params = cfg.get("parameters", {})

    video_folder = paths.get("video_folder", "data/videos/")
    output_folder = paths.get("output_folder", "data/output/")
    results_folder = paths.get("results_folder", "data/results/predicted")
    model_path = paths.get("model_path", "models/yolov8s/weights/best.pt")
    video_filename = paths.get("video_filename", "")

    if not video_filename:
        print("Missing video_filename in config.")
        return None

    mode = params.get("mode", "custom")
    model_type = params.get("model_type", cfg.get("model_type", "yolo"))
    confidence = float(params.get("confidence_threshold", 0.4))
    iou = float(params.get("iou_threshold", 0.5))
    grey_zone_size = float(params.get("grey_zone_size", 20.0))
    device = params.get("device", "cpu")
    show_window = bool(params.get("show_window", True))
    progress_every = int(params.get("progress_every_n_frames", 100))
    track_iou_threshold = float(params.get("track_iou_threshold", 0.3))
    track_max_lost = int(params.get("track_max_lost", 15))
    track_match_classes = bool(params.get("track_match_classes", True))
    rfdetr_box_format = params.get("rfdetr_box_format", "xyxy")
    rfdetr_box_normalized = params.get("rfdetr_box_normalized", "auto")

    model_name = _resolve_model_name(model_path, mode)
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    output_path = os.path.join(output_folder, f"output_{base_name}_{model_name}.mp4")

    cap = cv2.VideoCapture(os.path.join(video_folder, video_filename))
    ret, frame = cap.read()
    if not ret:
        print("Cannot open video.")
        return None

    use_interactive_lines = bool(params.get("use_interactive_lines", True))
    lines = []
    if use_interactive_lines:
        num_lines = int(params.get("num_lines", 1))
        try:
            num_lines = int(input("Kolik car chcete nakreslit? ") or num_lines)
        except ValueError:
            pass
        try:
            lines = Counter.select_lines_interactive(frame, num_lines)
        except cv2.error as exc:
            print(f"Interactive line selection failed ({exc}). Falling back to config lines.")

    if not lines:
        lines = params.get("lines", [])
        if not lines:
            print("Missing lines in config (parameters.lines).")
            return None
    for line in lines:
        print(f"{line['name']}: {line['start']} -> {line['end']}")

    counter = Counter(
        model_path=model_path,
        lines=lines,
        min_distance=grey_zone_size,
        confidence=confidence,
        iou=iou,
        device=device,
        pretrained=(mode == "pretrained"),
        model_type=model_type,
        track_iou_threshold=track_iou_threshold,
        track_max_lost=track_max_lost,
        track_match_classes=track_match_classes,
        rfdetr_box_format=rfdetr_box_format,
        rfdetr_box_normalized=rfdetr_box_normalized,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    print("Processing video...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        counter.process_frame(frame)
        counter.draw(frame)

        out.write(frame)
        if total_frames > 0:
            print(f"{frame_idx}/{total_frames}", end="\r", flush=True)
        else:
            print(f"{frame_idx}/?", end="\r", flush=True)
        if show_window:
            try:
                cv2.imshow("Counting", frame)
            except cv2.error as exc:
                print(f"OpenCV window disabled ({exc}). Continuing without display.")
                show_window = False
        elif progress_every > 0 and frame_idx % progress_every == 0:
            print(f"Processed {frame_idx} frames...")

        if show_window:
            try:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                show_window = False

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    results = counter.get_counts()
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    for line_name, counts in results.items():
        print(f"\n--- {line_name} ---")
        for class_id, name in DetectedObject.class_names.items():
            in_c = counts["in"].get(class_id, 0)
            out_c = counts["out"].get(class_id, 0)
            print(f"{name}: IN={in_c}, OUT={out_c}")
        print(f"TOTAL: IN={counts['total_in']}, OUT={counts['total_out']}")

    save_results_to_csv(results, video_filename, model_name, output_folder=results_folder)
    print(f"Video saved to: {output_path}")
    return results
