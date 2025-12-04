import cv2
import csv
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
video_path = "Videos/OxfordTownCentre/TownCentreXVID.mp4"
model_name = 'yolov8n.pt'
log_filename = 'occupancy_log.csv'

# --- FEATURE FLAGS ---
ENABLE_TRAJECTORY = True # Rasto Branco Permanente
MAX_OCCUPANCY = 20

# FIXED CONFIGURATIONS
line_y = 300
tracker_file = "my_tracker.yaml"
line_start = (0, line_y)
line_end = (1020, line_y)

# --- LOG SETUP (CSV) ---
if not os.path.exists(log_filename):
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Hour", "Person_ID", "Direction", "Occupancy", "X", "Y"])

# --- INITIALIZATION ---
model = YOLO(model_name)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error opening video '{video_path}'.")
    exit()

# --- STATE MANAGEMENT ---
ids_seen_bottom = set()
ids_seen_top = set()
status_green_ids = set()
status_cyan_ids = set()

# TRAJECTORY HISTORY GLOBAL
# Guardamos TODOS os pontos de TODOS os IDs para desenhar sempre
track_history = {}

count_in = 0
count_out = 0
occupancy = 0

print(f"🚀 Processing... Permanent Trajectories: ON")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (1020, 600))
    visual_frame = frame.copy()

    # Tracking
    try:
        results = model.track(frame, persist=True, classes=0, tracker=tracker_file, verbose=False)
    except Exception as e:
        results = model.track(frame, persist=True, classes=0, tracker="botsort.yaml", verbose=False)

    # --- DESENHO DAS TRAJETÓRIAS PERMANENTES ---
    # Desenhamos isto PRIMEIRO para ficar "por baixo" das caixas e da linha
    if ENABLE_TRAJECTORY:
        for track_id, points_list in track_history.items():
            if len(points_list) > 1:
                # Converter lista de pontos para formato numpy
                points = np.hstack(points_list).astype(np.int32).reshape((-1, 1, 2))
                # Desenhar linha branca (espessura fina para não tapar tudo)
                cv2.polylines(visual_frame, [points], isClosed=False, color=(255, 255, 255), thickness=1)

    # Draw Line (Por cima das trajetórias)
    cv2.line(visual_frame, line_start, line_end, (0, 255, 255), 2)
    cv2.putText(visual_frame, f"TOP ZONE", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
    cv2.putText(visual_frame, f"BOTTOM ZONE", (10, line_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 128), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().tolist()
        xyxy_boxes = results[0].boxes.xyxy.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box_xywh, box_xyxy, track_id in zip(boxes, xyxy_boxes, track_ids):
            x, y, w, h = box_xywh
            x1, y1, x2, y2 = map(int, box_xyxy)
            
            center_x_float = x
            center_y_float = y + (h / 3) 
            center_y_int = int(center_y_float)

            # --- ATUALIZAR TRAJETÓRIA ---
            if ENABLE_TRAJECTORY:
                if track_id not in track_history:
                    track_history[track_id] = []
                
                # Adicionar ponto atual (sem remover os antigos!)
                # Usamos o ponto dos pés para ser mais realista com o caminhar
                point = (int(x), int(y + (h/3)))
                track_history[track_id].append(point)
                
                # NOTA: Removi o "pop(0)", logo a linha cresce para sempre.

            box_color = (128, 128, 128) 

            # --- 1. UPDATE HISTORY ---
            if center_y_int > line_y:
                ids_seen_bottom.add(track_id)
            elif center_y_int < line_y:
                ids_seen_top.add(track_id)

            # --- 2. LOGIC CHECK ---
            if center_y_int > line_y and track_id in ids_seen_top:
                if track_id not in status_cyan_ids:
                    count_out += 1
                    occupancy -= 1
                    status_cyan_ids.add(track_id)
                    if track_id in status_green_ids: status_green_ids.remove(track_id)
                    
                    # Log OUT
                    now = datetime.now()
                    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    current_hour = now.strftime("%H")
                    log_x, log_y = round(center_x_float, 2), round(center_y_float, 2)
                    with open(log_filename, 'a', newline='') as f:
                        csv.writer(f).writerow([timestamp_str, current_hour, track_id, "OUT", occupancy, log_x, log_y])

            elif center_y_int < line_y and track_id in ids_seen_bottom:
                if track_id not in status_green_ids:
                    count_in += 1
                    occupancy += 1
                    status_green_ids.add(track_id)
                    if track_id in status_cyan_ids: status_cyan_ids.remove(track_id)

                    # Log IN
                    now = datetime.now()
                    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    current_hour = now.strftime("%H")
                    log_x, log_y = round(center_x_float, 2), round(center_y_float, 2)
                    with open(log_filename, 'a', newline='') as f:
                        csv.writer(f).writerow([timestamp_str, current_hour, track_id, "IN", occupancy, log_x, log_y])

            # --- 3. DETERMINE COLOR ---
            if track_id in status_cyan_ids:
                box_color = (255, 255, 0)
            elif track_id in status_green_ids:
                box_color = (50, 255, 50)
            elif center_y_int > line_y:
                box_color = (71, 99, 255)

            cv2.rectangle(visual_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(visual_frame, f"{track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # --- INFO PANEL ---
    occupancy_pct = (occupancy / MAX_OCCUPANCY) * 100 if MAX_OCCUPANCY > 0 else 0
    if occupancy_pct < 50:
        status_text = "LOW"
        status_color = (0, 255, 0)
    elif occupancy_pct < 85:
        status_text = "MODERATE"
        status_color = (0, 165, 255)
    else:
        status_text = "FULL"
        status_color = (0, 0, 255)

    cv2.rectangle(visual_frame, (20, 20), (350, 160), (0, 0, 0), -1)
    cv2.putText(visual_frame, f"IN: {count_in}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(visual_frame, f"OUT: {count_out}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(visual_frame, f"Occupancy: {occupancy}/{MAX_OCCUPANCY}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(visual_frame, f"Occupancy Status: {status_text}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    bar_x, bar_y, bar_w, bar_h = 30, 140, 300, 10
    cv2.rectangle(visual_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    fill_width = int(min(occupancy_pct, 100) / 100 * bar_w)
    fill_width = max(0, fill_width)
    cv2.rectangle(visual_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_h), status_color, -1)

    cv2.imshow("Store Analytics - Permanent Trajectory", visual_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()