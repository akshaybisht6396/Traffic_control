import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('test_traffic.mp4')

# ✅ FIXED: Coordinates remapped for 640x360 (your actual video size)
# Line sits right at the stop line visible in the footage
LINE_START = (0, 218)
LINE_END = (640, 276)
LINE_COLOR = (0, 0, 255)

TARGET_CLASSES = [2, 3, 5, 7]

if (LINE_END[0] - LINE_START[0]) != 0:
    SLOPE = (LINE_END[1] - LINE_START[1]) / (LINE_END[0] - LINE_START[0])
else:
    SLOPE = 0

violation_list = set()
track_history = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ✅ FIXED: Resize to 640x360 to match your actual video
    frame = cv2.resize(frame, (640, 360))

    results = model.track(frame, persist=True, verbose=False, classes=TARGET_CLASSES)

    cv2.line(frame, LINE_START, LINE_END, LINE_COLOR, 3)
    cv2.putText(frame, "STOP LINE", (LINE_START[0] + 10, LINE_START[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOR, 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        current_ids = set(ids)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box

            # Bottom-center = tire contact point
            cx, cy = int((x1 + x2) / 2), int(y2)

            # Line Y at this vehicle's X
            line_y_at_x = LINE_START[1] + SLOPE * (cx - LINE_START[0])

            # ✅ FIXED: Use CENTER-Y (not bottom) to catch cars already partially past line
            # Also use a small tolerance buffer (+10px) so fast cars aren't missed
            cy_center = int((y1 + y2) / 2)

            if track_id in track_history:
                prev_cy = track_history[track_id]

                # Crosses from above → below the line
                if prev_cy < line_y_at_x and cy_center >= line_y_at_x - 10:
                    if track_id not in violation_list:
                        violation_list.add(track_id)
                        print(f"🚨 VIOLATION! Vehicle #{track_id} crossed at frame")

            track_history[track_id] = cy_center

            # Draw boxes
            if track_id in violation_list:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(frame, f"VIOLATION #{track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        for k in [k for k in track_history if k not in current_ids]:
            del track_history[k]

    cv2.rectangle(frame, (0, 0), (320, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"VIOLATIONS: {len(violation_list)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("Red Line Violation Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()