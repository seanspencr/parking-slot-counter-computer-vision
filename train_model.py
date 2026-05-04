import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
import cv2
import os
import pickle
import matplotlib.pyplot as plt

positive_path = "./dataset/vehicles"
negative_path = "./dataset/non-vehicles"   # FIX 1: separate paths

def extract_hog(image):
    # Data is guaranteed 64x64x3 — no resize or shape check needed
    image = color.rgb2gray(image)
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

X, y = [], []

for fname in os.listdir(positive_path):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img = cv2.imread(os.path.join(positive_path, fname))  # FIX 2: os.path.join
    X.append(extract_hog(img))
    y.append(1)

for fname in os.listdir(negative_path):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img = cv2.imread(os.path.join(negative_path, fname))  # FIX 2: os.path.join
    X.append(extract_hog(img))
    y.append(0)

X, y = np.array(X), np.array(y)
print(f"Dataset: {sum(y==1)} positives, {sum(y==0)} negatives, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # FIX 3: stratify
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, max_iter=5000, random_state=42))
])
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy : {model.score(X_test,  y_test):.4f}")

# Learning curve
# train_sizes, train_scores, val_scores = learning_curve(
#     Pipeline([
#         ('scaler', StandardScaler()),
#         ('svm', LinearSVC(C=1.0, max_iter=5000, random_state=42))
#     ]),
#     X, y, cv=5,
#     train_sizes=np.linspace(0.1, 1.0, 10),
#     scoring='accuracy',
#     n_jobs=-1
# )

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

# plt.figure(figsize=(12, 6))
# plt.title("Learning Curve")
# plt.xlabel("Training Set Size")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.fill_between(train_sizes,
#     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
#     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
#     alpha=0.1, color='r')
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
# plt.fill_between(train_sizes,
#     np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
#     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
#     alpha=0.1, color='g')
# plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='g', label='Validation score')
# plt.legend(loc='best')
# plt.savefig('learning_curve.png', dpi=100, bbox_inches='tight')
# plt.show()

