import pickle
import numpy as np
import cv2
from skimage.transform import resize
from skimage.feature import hog
from skimage import color
from skimage.transform import resize


MASK_PATH       = './mask.jpg'
VIDEO_PATH      = '../parking_1920_1080_loop.mp4'
MODEL_PATH      = '../model-v2.pkl'


EMPTY     = True
NOT_EMPTY = False
STEP      = 30          # process every N frames
MODEL     = pickle.load(open(MODEL_PATH, "rb"))


def extract_hog(image):
    """Extract HOG exactly as in train_model.py"""
    # Resize to 64x64 for consistency with training data
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

def empty_or_not(spot_bgr):
    features = extract_hog(spot_bgr).reshape(1, -1)
    y_output = MODEL.predict(features)
    return EMPTY if y_output == 0 else NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w  = int(values[i, cv2.CC_STAT_WIDTH])
        h  = int(values[i, cv2.CC_STAT_HEIGHT])
        slots.append([x1, y1, w, h])
    return slots

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


mask = cv2.imread(MASK_PATH, 0)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
cap  = cv2.VideoCapture(VIDEO_PATH)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots        = get_parking_spots_bboxes(connected_components)
spots_status = [None] * len(spots)
diffs        = [0.0]  * len(spots)

previous_frame = None
frame_nmr      = 0

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# ── Main loop ─────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        previous_frame = None
        frame_nmr      = 0
        continue

    if frame_nmr % STEP == 0 and previous_frame is not None:
        
        crops = {}
        for spot_indx, (x1, y1, w, h) in enumerate(spots):
            spot_crop            = frame[y1:y1+h, x1:x1+w]
            prev_crop            = previous_frame[y1:y1+h, x1:x1+w]
            diffs[spot_indx]     = calc_diff(spot_crop, prev_crop)
            crops[spot_indx]     = spot_crop

        
        max_diff = np.amax(diffs)
        if max_diff > 0:
            arr_ = [j for j in np.argsort(diffs)
                    if diffs[j] / max_diff > 0.4]
        else:
            arr_ = []

        for spot_indx in arr_:
            spots_status[spot_indx] = empty_or_not(crops[spot_indx])

    elif frame_nmr % STEP == 0 and previous_frame is None:
        # First frame — classify all spots
        for spot_indx, (x1, y1, w, h) in enumerate(spots):
            spot_crop              = frame[y1:y1+h, x1:x1+w]
            spots_status[spot_indx] = empty_or_not(spot_crop)

    if frame_nmr % STEP == 0:
        previous_frame = frame.copy()

    
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

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()