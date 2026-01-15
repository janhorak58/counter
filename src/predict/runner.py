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
    confidence = float(params.get("confidence_threshold", 0.4))
    iou = float(params.get("iou_threshold", 0.5))
    grey_zone_size = float(params.get("grey_zone_size", 20.0))
    device = params.get("device", "cpu")

    model_name = _resolve_model_name(model_path, mode)
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    output_path = os.path.join(output_folder, f"output_{base_name}_{model_name}.mp4")

    cap = cv2.VideoCapture(os.path.join(video_folder, video_filename))
    ret, frame = cap.read()
    if not ret:
        print("Cannot open video.")
        return None

    num_lines = int(input("Kolik car chcete nakreslit? "))
    lines = Counter.select_lines_interactive(frame, num_lines)
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
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counter.process_frame(frame)
        counter.draw(frame)

        out.write(frame)
        cv2.imshow("Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

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
