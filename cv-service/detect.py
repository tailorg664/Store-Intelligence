from ultralytics import YOLO
videos_data =  {"vid1": {"name": "1.mp4", "path": "../data/1.mp4", "status": "pending"},
                "vid2": {"name": "2.mp4", "path": "../data/2.mp4", "status": "pending"},
                "vid3": {"name": "3.mp4", "path": "../data/3.mp4", "status": "pending"},
                "vid4": {"name": "4.mp4", "path": "../data/4.mp4", "status": "pending"},
                "vid5": {"name": "5.mp4", "path": "../data/5.mp4", "status": "pending"}}
last_positions = {}
track_age = {}
model = YOLO("yolo11n.pt")

results = model.predict(
    source=videos_data["vid1"]["path"],
    save=True,
    device = 0,  # Use GPU if available
    classes=[0],  # Person class
    stream = True,
    vid_stride = 4
)

for result in results:

    boxes = result.boxes

    if boxes is None:
        continue

    ids = boxes.id

    if ids is None:
        continue

    coords = boxes.xyxy
    confs = boxes.conf

    for person_id, box, conf in zip(ids, coords, confs):

        person_id = int(person_id)

        x1, y1, x2, y2 = box.tolist()

        width = x2 - x1
        height = y2 - y1

        # -----------------------------
        # Filter 1: Ignore tiny detections
        # -----------------------------
        if width < 30 or height < 60:
            continue

        # -----------------------------
        # Filter 2: Ignore low confidence
        # -----------------------------
        if conf < 0.5:
            continue

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # First time seeing this ID
        if person_id not in last_positions:
            last_positions[person_id] = (center_x, center_y)
            track_age[person_id] = 1
            continue

        prev_x, prev_y = last_positions[person_id]

        movement = abs(center_x - prev_x) + abs(center_y - prev_y)

        track_age[person_id] += 1

        last_positions[person_id] = (center_x, center_y)

        # -----------------------------
        # Filter 3: Seen long enough
        # -----------------------------
        if track_age[person_id] < 10:
            continue

        # -----------------------------
        # Filter 4: Ignore static objects
        # -----------------------------
        if movement < 5:
            continue
        print("DEBUG: Entered movement section")
        print(
            f"Person={person_id} "
            f"Position=({int(center_x)}, {int(center_y)}) "
            f"Movement={movement:.2f}",
            flush=True
        )
print("Detection completed")