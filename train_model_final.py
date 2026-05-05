import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import cv2
import os
import pickle

# ── Paths ─────────────────────────────────────────────────────────
positive_path = "./dataset-v2/vehicles"
negative_path = "./dataset-v2/non-vehicles"

# ── HOG extraction ────────────────────────────────────────────────
def extract_hog(image):
    image = color.rgb2gray(image)
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

# ── Augmentation (positives only) ────────────────────────────────
def augment(img):
    """
    Returns list of augmented variants of a single image.
    All operations preserve the 64x64x3 shape.
    """
    variants = []

    # 1. Original
    variants.append(img)

    # 2. Horizontal flip
    variants.append(cv2.flip(img, 1))

    # 3. Brightness up
    variants.append(cv2.convertScaleAbs(img, alpha=1.3, beta=20))

    # 4. Brightness down
    variants.append(cv2.convertScaleAbs(img, alpha=0.7, beta=-20))

    # 5. Slight rotation +10°
    M = cv2.getRotationMatrix2D((32, 32), 10, 1.0)
    variants.append(cv2.warpAffine(img, M, (64, 64)))

    # 6. Slight rotation -10°
    M = cv2.getRotationMatrix2D((32, 32), -10, 1.0)
    variants.append(cv2.warpAffine(img, M, (64, 64)))

    # 7. Gaussian blur (simulates out-of-focus / distance)
    variants.append(cv2.GaussianBlur(img, (3, 3), 0))

    # 8. Horizontal flip + brightness up (combination)
    flipped = cv2.flip(img, 1)
    variants.append(cv2.convertScaleAbs(flipped, alpha=1.3, beta=20))

    return variants

# ── Load dataset ──────────────────────────────────────────────────
X, y = [], []

print("Loading positives (with augmentation)...")
pos_original = 0
for fname in os.listdir(positive_path):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img = cv2.imread(os.path.join(positive_path, fname))
    if img is None:
        continue
    
    if img.shape[:2] != (64, 64):
        img = cv2.resize(img, (64, 64))
        
    pos_original += 1
    for variant in augment(img):       # each image → 8 variants
        X.append(extract_hog(variant))
        y.append(1)

print(f"  {pos_original} original → {sum(1 for label in y if label == 1)} after augmentation")

print("Loading negatives (no augmentation)...")
neg_start = len(y)
for fname in os.listdir(negative_path):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img = cv2.imread(os.path.join(negative_path, fname))
    if img is None:
        continue
    
    if img.shape[:2] != (64, 64):
        img = cv2.resize(img, (64, 64))
    
    X.append(extract_hog(img))
    y.append(0)

print(f"  {len(y) - neg_start} negatives loaded")

X, y = np.array(X), np.array(y)
n_pos = sum(y == 1)
n_neg = sum(y == 0)
print(f"\nDataset: {n_pos} positives, {n_neg} negatives, {X.shape[1]} features")

# ── Warn if still imbalanced ──────────────────────────────────────
ratio = n_neg / n_pos if n_pos > 0 else float('inf')
if ratio > 3:
    print(f"Warning: negatives are {ratio:.1f}x positives — consider adding more positive images")

# ── Train / test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── Train ─────────────────────────────────────────────────────────
print("Training...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])
model.fit(X_train, y_train)

print(f"\nTrain Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy : {model.score(X_test,  y_test):.4f}")

# ── Save ──────────────────────────────────────────────────────────
with open("model-v2.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to model-v2.pkl")