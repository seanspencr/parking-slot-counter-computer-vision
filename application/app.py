import pickle, threading, time
import numpy as np
import cv2
from skimage.feature import hog
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────
MASK_PATH  = './mask.jpg'
CAMERA_SRC = "http://192.168.1.7:8080/video"                        # webcam; swap for RTSP URL
MODEL_PATH = './model-v2.pkl'
STEP       = 30

EMPTY     = True
NOT_EMPTY = False
MODEL     = pickle.load(open(MODEL_PATH, "rb"))

# ── Shared state (thread-safe with a lock) ────────────────────────
state_lock   = threading.Lock()
shared_state = {"available": 0, "total": 0, "last_update": None}

frame_lock   = threading.Lock()
latest_frame = None 

# ── HOG + model (unchanged from your original) ───────────────────
def extract_hog(image):
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys')

def empty_or_not(spot_bgr):
    features = extract_hog(spot_bgr).reshape(1, -1)
    return EMPTY if MODEL.predict(features)[0] == 0 else NOT_EMPTY

def get_parking_spots_bboxes(cc):
    total, _, values, _ = cc
    return [[int(values[i, cv2.CC_STAT_LEFT]),  int(values[i, cv2.CC_STAT_TOP]),
             int(values[i, cv2.CC_STAT_WIDTH]), int(values[i, cv2.CC_STAT_HEIGHT])]
            for i in range(1, total)]

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# ── Background camera thread ──────────────────────────────────────
def camera_loop():

    global latest_frame

    mask = cv2.imread(MASK_PATH, 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    spots = get_parking_spots_bboxes(
        cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    )

    spots_status = [None] * len(spots)
    previous_frame = None
    frame_nmr = 0

    cap = cv2.VideoCapture(CAMERA_SRC)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {CAMERA_SRC}")

    while True:
        ret, frame = cap.read()
        if not ret:
            # For webcam/live stream, retry instead of rewinding
            time.sleep(0.1)
            continue

        if frame_nmr % STEP == 0 and previous_frame is not None:
            diffs = []
            crops = {}
            for i, (x1, y1, w, h) in enumerate(spots):
                crop = frame[y1:y1+h, x1:x1+w]
                prev = previous_frame[y1:y1+h, x1:x1+w]
                diffs.append(calc_diff(crop, prev))
                crops[i] = crop

            max_diff = np.amax(diffs)
            changed = ([j for j in np.argsort(diffs)
                        if diffs[j] / max_diff > 0.4]
                       if max_diff > 0 else [])

            for i in changed:
                spots_status[i] = empty_or_not(crops[i])

        elif frame_nmr % STEP == 0 and previous_frame is None:
            for i, (x1, y1, w, h) in enumerate(spots):
                spots_status[i] = empty_or_not(frame[y1:y1+h, x1:x1+w])

        if frame_nmr % STEP == 0:
            previous_frame = frame.copy()
            available = sum(1 for s in spots_status if s is True)
            with state_lock:
                shared_state["available"]   = available
                shared_state["total"]       = len(spots)
                shared_state["last_update"] = time.time()


        # upudatae latest frame
        for spot_indx, (x1, y1, w, h) in enumerate(spots):
            status = spots_status[spot_indx]
            color  = (0, 255, 0) if status else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

        available = sum(1 for s in spots_status if s is True)
        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame,
                    f'Available spots: {available} / {len(spots)}',
                    (100, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        

        _, jpeg = cv2.imencode('.jpg', frame)
        with frame_lock:
            latest_frame = jpeg.tobytes()

        frame_nmr += 1


def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        # MJPEG format: each frame is wrapped in multipart boundaries
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )
        time.sleep(0.03)  # ~30fps cap


# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

@app.get("/slots")
def get_slots():
    with state_lock:
        return {
            "total":       shared_state["total"],
            "available":   shared_state["available"],
            "occupied":    shared_state["total"] - shared_state["available"],
            "last_update": shared_state["last_update"],
        }
    

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )