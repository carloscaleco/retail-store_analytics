import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
video_path = "Videos/OxfordTownCentre/TownCentreXVID.mp4"
model_name = 'yolov8n.pt'
log_filename = 'occupancy_log.csv'

# Counting Line Position
line_y = 300
line_start = (0, line_y)
line_end = (1020, line_y)

# --- LOG SETUP (CSV) ---
# We verify if file exists to write headers
if not os.path.exists(log_filename):
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Added "Direction" and "Occupancy" columns
        writer.writerow(["Timestamp", "Hour", "Person_ID", "Direction", "Occupancy", "X", "Y"])
    print(f"📄 Log file created: {log_filename}")
else:
    print(f"📄 Using existing log: {log_filename}")

# --- INITIALIZATION ---
model = YOLO(model_name)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error opening video '{video_path}'.")
    exit()

# --- STATE MANAGEMENT ---
# Candidates
ids_bottom_zone = set() # Seen in bottom (Candidates for IN)
ids_top_zone = set()    # Seen in top (Candidates for OUT)

# Completed Actions (to prevent double counting the same action)
counted_in_ids = set()
counted_out_ids = set()

# Counters
count_in = 0
count_out = 0
occupancy = 0

print("🚀 Processing... Logic: IN (Bottom->Top) | OUT (Top->Bottom)")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (1020, 600))
    visual_frame = frame.copy()

    # Tracking
    results = model.track(frame, persist=True, classes=0, tracker="botsort.yaml", verbose=False)

    # Draw Line
    cv2.line(visual_frame, line_start, line_end, (0, 255, 255), 2)
    
    # Zone Labels
    cv2.putText(visual_frame, "TOP ZONE (Exit Candidate)", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(visual_frame, "BOTTOM ZONE (Entry Candidate)", (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().tolist()
        xyxy_boxes = results[0].boxes.xyxy.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box_xywh, box_xyxy, track_id in zip(boxes, xyxy_boxes, track_ids):
            x, y, w, h = box_xywh
            x1, y1, x2, y2 = map(int, box_xyxy)
            
            # --- COORDINATES ---
            center_x_float = x
            center_y_float = y + (h / 3) 
            center_y_int = int(center_y_float)

            # Default Color (Grey = Ignored/Neutral)
            box_color = (128, 128, 128)

            # --- LOGIC 1: ENTRY (Bottom -> Top) ---
            # If not yet counted as IN
            if track_id not in counted_in_ids:
                # If currently in Bottom Zone
                if center_y_int > line_y:
                    ids_bottom_zone.add(track_id)
                    box_color = (0, 0, 255) # Red (Candidate IN)
                
                # If currently in Top Zone AND was seen in Bottom
                elif center_y_int < line_y and track_id in ids_bottom_zone:
                    count_in += 1
                    occupancy += 1
                    counted_in_ids.add(track_id)
                    
                    # LOG ENTRY
                    now = datetime.now()
                    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    current_hour = now.strftime("%H")
                    log_x = round(center_x_float, 2)
                    log_y = round(center_y_float, 2)

                    print(f"✅ IN: ID {track_id}. Occupancy: {occupancy}")
                    
                    with open(log_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp_str, current_hour, track_id, "IN", occupancy, log_x, log_y])

            # --- LOGIC 2: EXIT (Top -> Bottom) ---
            # If not yet counted as OUT
            if track_id not in counted_out_ids:
                # If currently in Top Zone
                if center_y_int < line_y:
                    ids_top_zone.add(track_id)
                    # Use Blue for Exit Candidates (unless they just entered, then keep Green briefly)
                    if track_id not in counted_in_ids: 
                        box_color = (255, 0, 0) # Blue (Candidate OUT)
                
                # If currently in Bottom Zone AND was seen in Top
                elif center_y_int > line_y and track_id in ids_top_zone:
                    count_out += 1
                    occupancy -= 1 # Decrease occupancy
                    counted_out_ids.add(track_id)

                    # LOG EXIT
                    now = datetime.now()
                    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    current_hour = now.strftime("%H")
                    log_x = round(center_x_float, 2)
                    log_y = round(center_y_float, 2)

                    print(f"🔻 OUT: ID {track_id}. Occupancy: {occupancy}")

                    with open(log_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp_str, current_hour, track_id, "OUT", occupancy, log_x, log_y])

            # --- VISUAL FEEDBACK (Green overrides others if action just happened) ---
            # Note: In a real scenario, you might want a timer to clear 'Green'. 
            # Here, once counted, they stay Green to show they are "Processed".
            if track_id in counted_in_ids and center_y_int < line_y:
                 box_color = (0, 255, 0) # Green (Successfully Entered)
            
            if track_id in counted_out_ids and center_y_int > line_y:
                 box_color = (0, 255, 0) # Green (Successfully Exited)

            # Draw Box
            cv2.rectangle(visual_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(visual_frame, f"{track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # --- INFO PANEL ---
    # Background
    cv2.rectangle(visual_frame, (20, 20), (350, 110), (0, 0, 0), -1)
    
    # Text Stats
    cv2.putText(visual_frame, f"IN: {count_in}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(visual_frame, f"OUT: {count_out}", (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Occupancy (White)
    cv2.putText(visual_frame, f"OCCUPANCY: {occupancy}", (30, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Store Occupancy Tracking", visual_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()