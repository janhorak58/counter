# loads first frame of video and lets te user choos line visually. then prints the coordinates
import cv2
import sys
from pathlib import Path
from typing import Tuple
from counter.core.types import LineCoords

def choose_line(video_path: Path) -> LineCoords:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        sys.exit(1)

    line_coords = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            line_coords.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Choose Line", frame)
            if len(line_coords) == 2:
                cv2.line(frame, line_coords[0], line_coords[1], (255, 0, 0), 2)
                cv2.imshow("Choose Line", frame)

    cv2.imshow("Choose Line", frame)
    cv2.setMouseCallback("Choose Line", click_event)

    print("Click two points to define the counting line.")
    while True:
        cv2.imshow("Choose Line", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            print("Exiting without choosing a line.")
            sys.exit(0)
        if len(line_coords) == 2:
            print("Line chosen.")
            # wait for "q" to confirm
            print("Press 'q' to confirm the line.")
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == ord('q'):
                    break
            break
    cv2.destroyAllWindows()
    cap.release()

    x1, y1 = line_coords[0]
    x2, y2 = line_coords[1]
    return (float(x1), float(y1), float(x2), float(y2))

if __name__ == "__main__":
    print("Choose a video file to select a counting line.")
    if len(sys.argv) != 2:
        print("Usage: python choose_line.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Video file does not exist: {video_path}")
        sys.exit(1)

    line = choose_line(video_path)
    print(f"Chosen line coordinates: {line}")