import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Parking slot coordinates
slot_coordinates = [
    [[7, 161], [54, 5], [215, 14], [131, 175]],
    [[216, 14], [361, 17], [303, 184], [131, 178]],
    [[356, 17], [548, 24], [486, 187], [305, 187]],
    [[547, 26], [710, 29], [653, 192], [488, 187]],
    [[842, 30], [712, 27], [656, 193], [795, 200]],
    [[839, 32], [979, 35], [955, 195], [796, 202]],
    [[1146, 45], [980, 36], [955, 196], [1138, 211]],
    [[1138, 211], [1339, 215], [1333, 54], [1143, 46]],
    [[1336, 46], [1476, 55], [1506, 214], [1345, 216]],
    [[1680, 63], [1734, 215], [1510, 216], [1482, 55]],
    [[1750, 533], [1755, 313], [1579, 307], [1613, 524]],
    [[1573, 307], [1371, 304], [1419, 526], [1612, 523]],
    [[1385, 303], [1417, 523], [1181, 516], [1175, 304]],
    [[1175, 304], [1175, 516], [975, 511], [992, 300]],
    [[818, 296], [789, 507], [975, 510], [992, 299]],
    [[818, 297], [784, 505], [575, 501], [634, 299]],
    [[634, 299], [575, 499], [402, 494], [470, 293]],
    [[470, 293], [399, 494], [177, 488], [261, 296]],
    [[252, 293], [181, 486], [4, 480], [33, 288]]
]

# Driveway and car coordinates
driveway_coordinates = [[33, 282], [36, 180], [1578, 225], [1581, 301]]
car_in_driveway = [[938, 215], [1157, 223], [1157, 292], [939, 292]]

# Function to compute centroid of a given quadrilateral
def get_centroid(coords):
    x = sum(p[0] for p in coords) / len(coords)
    y = sum(p[1] for p in coords) / len(coords)
    return (int(x), int(y))

# Function to find the nearest empty parking slot and its slot number
def find_nearest_empty_slot(car_coords, tracked_slots):
    car_center = get_centroid(car_coords)
    min_distance = float('inf')
    nearest_slot = None
    nearest_slot_index = None

    for idx, slot in enumerate(tracked_slots):
        if not slot['occupied']:
            slot_center = get_centroid(slot['coordinates'])
            distance = np.linalg.norm(np.array(car_center) - np.array(slot_center))
            if distance < min_distance:
                min_distance = distance
                nearest_slot = slot_center
                nearest_slot_index = idx + 1  # Since slots are indexed from 1 in display
    
    return car_center, nearest_slot, nearest_slot_index

# Function to track parking slot occupancy
def track_parking_slots(image, model, slot_coordinates):
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    occupied_slots = 0
    empty_slots = len(slot_coordinates)    
    tracked_slots = []
    
    for idx, coords in enumerate(slot_coordinates, start=1):  
        pts = np.array(coords, dtype=np.int32)  
        pts = pts.reshape((-1, 1, 2))

        slot_occupied = False
        for box, class_id in zip(boxes, class_ids):
            if int(class_id) == 2 or int(class_id) == 3:  # Cars and motorbikes (YOLO class IDs)
                car_x1, car_y1, car_x2, car_y2 = box
                vehicle_center = ((car_x1 + car_x2) / 2, (car_y1 + car_y2) / 2)
                distance = cv2.pointPolygonTest(pts, vehicle_center, False)
                if distance >= 0:
                    slot_occupied = True
                    break

        color = (0, 0, 255) if slot_occupied else (0, 255, 0)  # Red if occupied, green if empty
        cv2.fillPoly(image, [pts], color)

        if slot_occupied:
            occupied_slots += 1
            empty_slots -= 1

        tracked_slots.append({
            "coordinates": coords,
            "occupied": slot_occupied
        })

        M = cv2.moments(pts)  
        if M["m00"] != 0:  
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(image, f"{idx}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   
    return tracked_slots, occupied_slots, empty_slots, image

# Function to resize the image to fit the screen
def resize_image(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size)

# Load the image
image_path = "p1.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    tracked_slots, occupied_slots, empty_slots, result_image = track_parking_slots(image, model, slot_coordinates)
    car_pos, nearest_slot, nearest_slot_index = find_nearest_empty_slot(car_in_driveway, tracked_slots)
    
    if nearest_slot:
        cv2.line(result_image, car_pos, nearest_slot, (255, 255, 0), 3)
        cv2.putText(result_image, f"Navigate to Slot {nearest_slot_index}", nearest_slot, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print(f"Occupied Slots: {occupied_slots}")
    print(f"Empty Slots: {empty_slots}")
    print(f"Navigation Slot Number: {nearest_slot_index}")

    # Display the text on image
    cv2.putText(result_image, f"Occupied Slots: {occupied_slots}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_image, f"Empty Slots: {empty_slots}", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_image, f"Navigate to Slot {nearest_slot_index}", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Resize the image to fit the screen
    resized_image = resize_image(result_image)
    
    # Display the result
    cv2.imshow("Parking Slot Detection", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
