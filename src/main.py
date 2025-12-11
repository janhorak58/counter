import cv2
from src.core.Counter import Counter
from src.core.DetectedObject import DetectedObject
import yaml
import os

CONFIG_PATH = 'conf.yaml'

try:
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Konfigurační soubor '{CONFIG_PATH}' nebyl nalezen.")
    exit(1)
paths = config.get('paths', {})
VIDEO_FOLDER = paths.get('video_folder', '..\\data\\videos\\')
OUTPUT_FOLDER = paths.get('output_folder', '..\\data\\output\\')
MODEL_PATH = paths.get('model_path', f"models/yolov8s/weights/best.pt")

parameters = config.get('parameters', {})
CONFIDENCE_THRESHOLD = float(parameters.get('confidence_threshold', 0.4))
IOU_THRESHOLD = float(parameters.get('iou_threshold', 0.5))
IOU_THRESHOLD = float(parameters.get('iou_threshold', 0.5))
GREY_ZONE_SIZE = float(parameters.get('grey_zone_size', 20.0))
DEVICE = parameters.get('device', 'cpu')  # 'cpu' nebo "0"


def main(*args, **kwargs):
    if len(args) >= 1:
        video_filename = args[0]
    else:
        video_filename = 'vid4.mp4'
    output_path = OUTPUT_FOLDER + f"output_{video_filename}"

    cap = cv2.VideoCapture(VIDEO_FOLDER + video_filename)
    ret, frame = cap.read()
    if not ret:
        print("Nelze načíst video!")
        return
    
    # Interaktivní výběr VÍCE čar
    num_lines = int(input("Kolik čar chcete nakreslit? "))
    lines = Counter.select_lines_interactive(frame, num_lines)
    
    for line in lines:
        print(f"{line['name']}: {line['start']} -> {line['end']}")
    
    # Inicializace counteru s více čarami
    counter = Counter(
        model_path=MODEL_PATH,
        lines=lines,
        min_distance=GREY_ZONE_SIZE,
        confidence=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE
    )
    
    # Příprava výstupu
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    
    print("Zpracovávám video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        counter.process_frame(frame)
        counter.draw(frame)
        
        out.write(frame)
        cv2.imshow("Counting", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Výsledky pro všechny čáry
    results = counter.get_counts()
    print("\n" + "="*50)
    print("FINÁLNÍ VÝSLEDKY:")
    print("="*50)
    
    for line_name, counts in results.items():
        print(f"\n--- {line_name} ---")
        for class_id, name in DetectedObject.class_names.items():
            in_c = counts['in'].get(class_id, 0)
            out_c = counts['out'].get(class_id, 0)
            print(f"{name}: IN={in_c}, OUT={out_c}")
        print(f"TOTAL: IN={counts['total_in']}, OUT={counts['total_out']}")
    
    print(f"\nVideo uloženo: {output_path}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])