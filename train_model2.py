import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────
POS_DIR = "./dataset/vehicles"
NEG_DIR = "./dataset/non-vehicles"

# ── COLOR HOG (all 3 channels) ───────────────────────────────────
def extract_features(img_bgr):
    """
    Extract HOG from all 3 channels separately + color histogram.
    Much richer than grayscale HOG alone.
    """
    features = []

    # HOG per channel (B, G, R)
    for ch in cv2.split(img_bgr):
        ch_norm = ch.astype(np.float32) / 255.0
        feat = hog(
            ch_norm,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(feat)

    # Color histogram per channel (captures car color patterns)
    for ch in cv2.split(img_bgr):
        hist = cv2.calcHist([ch], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    return np.concatenate(features)

# ── AUGMENTATION ─────────────────────────────────────────────────
def augment(img):
    variants = [img]
    variants.append(cv2.flip(img, 1))                                        # horizontal flip
    variants.append(cv2.convertScaleAbs(img, alpha=1.3, beta=20))            # brighter
    variants.append(cv2.convertScaleAbs(img, alpha=0.7, beta=-20))           # darker

    # Slight rotation
    M10 = cv2.getRotationMatrix2D((32, 32), 10, 1.0)
    M_10 = cv2.getRotationMatrix2D((32, 32), -10, 1.0)
    variants.append(cv2.warpAffine(img, M10,  (64, 64)))
    variants.append(cv2.warpAffine(img, M_10, (64, 64)))

    return variants

# ── LOAD ─────────────────────────────────────────────────────────
def load_dir(directory, label, use_augment=False):
    X, y = [], []
    files = [f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in files:
        img = cv2.imread(os.path.join(directory, fname))
        if img is None:
            continue
        if img.shape != (64, 64, 3):
            img = cv2.resize(img, (64, 64))   # safety net

        imgs = augment(img) if use_augment else [img]
        for v in imgs:
            X.append(extract_features(v))
            y.append(label)

    return X, y

print("Loading dataset...")
X_pos, y_pos = load_dir(POS_DIR, label=1, use_augment=True)
X_neg, y_neg = load_dir(NEG_DIR, label=0, use_augment=False)

print(f"Positives : {len(X_pos)}")
print(f"Negatives : {len(X_neg)}")

# ── BALANCE ──────────────────────────────────────────────────────
n_min = min(len(X_pos), len(X_neg))
if len(X_pos) > len(X_neg):
    X_pos, y_pos = resample(X_pos, y_pos, n_samples=n_min, random_state=42)
elif len(X_neg) > len(X_pos):
    X_neg, y_neg = resample(X_neg, y_neg, n_samples=n_min, random_state=42)

X = np.array(X_pos + X_neg)
y = np.array(y_pos + y_neg)
print(f"Balanced  : {sum(y==1)} pos | {sum(y==0)} neg")

# ── SPLIT ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── TRAIN ────────────────────────────────────────────────────────
print("\nTraining...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced'))
])
model.fit(X_train, y_train)

# ── EVALUATE ─────────────────────────────────────────────────────
train_acc = model.score(X_train, y_train)
test_acc  = model.score(X_test,  y_test)
y_pred    = model.predict(X_test)

print(f"\nTrain Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Overfit Gap    : {train_acc - test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Car", "Car"]))

# ── CROSS VALIDATION ─────────────────────────────────────────────
print("5-Fold Cross Validation...")
cv_scores = cross_val_score(
    model, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
print(f"CV Scores : {np.round(cv_scores, 4)}")
print(f"CV Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── CONFUSION MATRIX ─────────────────────────────────────────────
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Not Car", "Car"]
).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── DATASET SANITY CHECK ─────────────────────────────────────────
print("\n── Sanity Check ──────────────────────────────")
print(f"Feature vector length : {X.shape[1]}")
print(f"Pixel value range     : {X.min():.2f} – {X.max():.2f}")
print(f"Any NaN in features   : {np.isnan(X).any()}")