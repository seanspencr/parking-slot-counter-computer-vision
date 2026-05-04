import cv2
import pickle
from skimage.feature import hog
from skimage import color
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

def extract_hog(image):
    # Data guaranteed 64x64x3 — just convert to gray, no resize needed
    image = color.rgb2gray(image)
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

def sliding_window(image, step_size, window_size=(64, 64)):
    h, w = image.shape[:2]
    for y in range(0, h - window_size[1], step_size):
        for x in range(0, w - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def image_pyramid(image, scale=1.25, min_size=(64, 64)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        if w < min_size[0] or h < min_size[1]:
            break
        image = cv2.resize(image, (w, h))
        yield image

def non_max_suppression(boxes, scores, overlap_thresh=0.15):
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w    = np.maximum(0.0, xx2 - xx1 + 1)
        h    = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]
        order   = order[np.where(overlap <= overlap_thresh)[0] + 1]
    return keep

def detect_and_count(image, model,
                     scale=1.25,
                     step=32,              # FIX 2: was 16, bigger step = less redundancy
                     window=(64, 64),
                     threshold=1.0,        # FIX 4: was 0.5, LinearSVC scores need higher bar
                     overlap_thresh=0.15): # FIX 3: was 0.35, suppress more aggressively

    detections, scores = [], []
    scale_factor = 1.0

    for resized in image_pyramid(image, scale=scale):
        for (x, y, crop) in sliding_window(resized, step, window):
            if crop.shape[:2] != (window[1], window[0]):
                continue

            features = extract_hog(crop).reshape(1, -1)
            score    = model.decision_function(features)[0]

            if score > threshold:
                # FIX 1: was (x * scale_factor) which grows coords — wrong
                # pyramid shrinks image, so coords must scale UP by dividing
                detections.append([
                    int(x / scale_factor),
                    int(y / scale_factor),
                    int((x + window[0]) / scale_factor),
                    int((y + window[1]) / scale_factor)
                ])
                scores.append(score)

        scale_factor /= scale  # FIX 1 cont: track inverse of shrink factor

    keep        = non_max_suppression(detections, scores, overlap_thresh)
    final_boxes = [detections[i] for i in keep]
    return final_boxes, len(final_boxes)


# ── RUN ──────────────────────────────────────────────────────────
image = cv2.imread("please.jpg")
boxes, count = detect_and_count(image, model)
print(f"Cars detected: {count}")

for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(image, f"Count: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Detections", image)
cv2.waitKey(0)