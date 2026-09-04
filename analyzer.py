import cv2
import numpy as np
import base64
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

def process_puttu_and_banana(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image format"}

    h_img, w_img, _ = img.shape

    # 1. Detect Banana with YOLO
    results = model(img, verbose=False)[0]
    banana_boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if results.names[cls_id] == "banana":
            banana_boxes.append(box.xyxy[0].cpu().numpy())

    if not banana_boxes:
        return {"error": "No banana detected in image!"}

    # 2. Create a Precise Banana Mask using HSV Color Thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Restrict yellow mask strictly within YOLO bounding boxes to avoid false positives
    yolo_roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    total_banana_vol_px = 0

    for bbox in banana_boxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(yolo_roi_mask, (x1, y1), (x2, y2), 255, -1)
        
        b_w, b_h = x2 - x1, y2 - y1
        r_b = min(b_w, b_h) / 2.0
        len_b = max(b_w, b_h)
        total_banana_vol_px += np.pi * (r_b ** 2) * len_b

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, "Banana", (x1, max(y1 - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    precise_banana_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=yolo_roi_mask)

    # 3. Detect Puttu using Grayscale Thresholding (Handles Steel Plates)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu thresholding isolates bright white puttu from darker plate shadows
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Erase banana pixels using precise HSV mask (prevents box clipping)
    thresh[precise_banana_mask > 0] = 0

    # Clean up small crumbles with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Find All Puttu Contours (Handles Broken Pieces)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny scattered rice crumbles
    puttu_cnts = [c for c in cnts if cv2.contourArea(c) > 3000]

    if not puttu_cnts:
        return {"error": "Could not locate Puttu. Try placing on a non-reflective dark plate."}

    total_puttu_vol_px = 0
    for c in puttu_cnts:
        rect = cv2.minAreaRect(c)
        (cx, cy), (w_p, h_p), _ = rect
        
        r_p = min(w_p, h_p) / 2.0
        h_p_val = max(w_p, h_p)
        total_puttu_vol_px += np.pi * (r_p ** 2) * h_p_val

        # Draw box around every detected puttu piece
        box_pts = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(img, [box_pts], 0, (0, 255, 0), 2)

    # 5. Volume Ratio
    volume_ratio = total_banana_vol_px / total_puttu_vol_px
    TARGET_RATIO = 0.35  # Target: Banana volume ~35% of puttu volume
    sufficiency_pct = min(int((volume_ratio / TARGET_RATIO) * 100), 1000)

    is_enough = volume_ratio >= TARGET_RATIO
    status_text = "ENOUGH BANANA!" if is_enough else "NEED MORE BANANA!"
    status_color = (0, 255, 0) if is_enough else (0, 0, 255)

    # Draw Banner
    cv2.rectangle(img, (0, 0), (w_img, 50), (0, 0, 0), -1)
    cv2.putText(img, f"{status_text} ({sufficiency_pct}% of target ratio)", 
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": status_text,
        "is_enough": is_enough,
        "sufficiency_percentage": sufficiency_pct,
        "volume_ratio": round(volume_ratio, 3),
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }
