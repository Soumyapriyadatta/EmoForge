import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
TRAIN_DIR = 'face/data/archive/train'
TEST_DIR  = 'face/data/archive/test'
SAVE_DIR  = 'face/data/processed'
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────
# EMOTION LABELS
# ─────────────────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ─────────────────────────────────────────
# LOAD IMAGES
# ─────────────────────────────────────────
def load_images(data_dir):
    images = []
    labels = []
    for label_idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: {emotion_path} not found")
            continue
        for img_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label_idx)
    return np.array(images), np.array(labels)

print("Loading training images...")
X_train, y_train = load_images(TRAIN_DIR)
print(f"Training set: {X_train.shape} images")

print("Loading test images...")
X_test, y_test = load_images(TEST_DIR)
print(f"Test set: {X_test.shape} images")

# ─────────────────────────────────────────
# NORMALIZE
# ─────────────────────────────────────────
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Reshape for CNN input (add channel dimension)
X_train = X_train.reshape(-1, 48, 48, 1)
X_test  = X_test.reshape(-1, 48, 48, 1)

print(f"\nAfter normalization:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# ─────────────────────────────────────────
# CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\nClass distribution (training set):")
for i, emotion in enumerate(EMOTIONS):
    count = np.sum(y_train == i)
    print(f"  {emotion}: {count} images")

# ─────────────────────────────────────────
# PLOT CLASS DISTRIBUTION
# ─────────────────────────────────────────
plt.figure(figsize=(10, 5))
counts = [np.sum(y_train == i) for i in range(len(EMOTIONS))]
plt.bar(EMOTIONS, counts, color='steelblue')
plt.title('FER2013 — Training Set Class Distribution')
plt.xlabel('Emotion')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/fer2013_class_distribution.png')
plt.show()
print("Class distribution plot saved.")

# ─────────────────────────────────────────
# PLOT SAMPLE IMAGES
# ─────────────────────────────────────────
plt.figure(figsize=(14, 4))
for i, emotion in enumerate(EMOTIONS):
    idx = np.where(y_train == i)[0][0]
    plt.subplot(1, 7, i+1)
    plt.imshow(X_train[idx].reshape(48, 48), cmap='gray')
    plt.title(emotion)
    plt.axis('off')
plt.suptitle('FER2013 — Sample Images per Emotion')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/fer2013_sample_images.png')
plt.show()
print("Sample images plot saved.")

# ─────────────────────────────────────────
# SAVE PROCESSED DATA
# ─────────────────────────────────────────
print("\nSaving processed data...")
np.save(f'{SAVE_DIR}/X_train.npy', X_train)
np.save(f'{SAVE_DIR}/y_train.npy', y_train)
np.save(f'{SAVE_DIR}/X_test.npy',  X_test)
np.save(f'{SAVE_DIR}/y_test.npy',  y_test)
print("Saved successfully:")
print(f"  {SAVE_DIR}/X_train.npy")
print(f"  {SAVE_DIR}/y_train.npy")
print(f"  {SAVE_DIR}/X_test.npy")
print(f"  {SAVE_DIR}/y_test.npy")
print("\nFER2013 preprocessing complete! ✅")