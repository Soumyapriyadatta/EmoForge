import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
PROCESSED_DIR = 'face/data/processed'
SAVE_DIR      = 'face/data/processed'

# ─────────────────────────────────────────
# LOAD PROCESSED DATA
# ─────────────────────────────────────────
print("Loading processed FER2013 data...")
X_train = np.load(f'{PROCESSED_DIR}/X_train.npy')
y_train = np.load(f'{PROCESSED_DIR}/y_train.npy')

print(f"Original training set: {X_train.shape}")

# ─────────────────────────────────────────
# EMOTIONS
# ─────────────────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ─────────────────────────────────────────
# CHECK CLASS DISTRIBUTION BEFORE
# ─────────────────────────────────────────
print("\nClass distribution BEFORE augmentation:")
for i, emotion in enumerate(EMOTIONS):
    count = np.sum(y_train == i)
    print(f"  {emotion}: {count} images")

# ─────────────────────────────────────────
# AUGMENTATION SETTINGS
# ─────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# ─────────────────────────────────────────
# AUGMENT MINORITY CLASSES
# Target: bring all classes to at least 3000 samples
# ─────────────────────────────────────────
TARGET = 3000

X_aug_list = [X_train]
y_aug_list = [y_train]

for class_idx, emotion in enumerate(EMOTIONS):
    class_images = X_train[y_train == class_idx]
    current_count = len(class_images)

    if current_count >= TARGET:
        print(f"{emotion}: {current_count} — no augmentation needed")
        continue

    needed = TARGET - current_count
    print(f"{emotion}: {current_count} → generating {needed} new samples...")

    augmented = []
    gen = datagen.flow(class_images, batch_size=32, shuffle=True)
    while len(augmented) < needed:
        batch = next(gen)
        augmented.extend(batch)

    augmented = np.array(augmented[:needed])
    aug_labels = np.full(needed, class_idx)

    X_aug_list.append(augmented)
    y_aug_list.append(aug_labels)

# ─────────────────────────────────────────
# COMBINE ORIGINAL + AUGMENTED
# ─────────────────────────────────────────
X_final = np.concatenate(X_aug_list, axis=0)
y_final = np.concatenate(y_aug_list, axis=0)

print(f"\nFinal training set: {X_final.shape}")

# ─────────────────────────────────────────
# CHECK CLASS DISTRIBUTION AFTER
# ─────────────────────────────────────────
print("\nClass distribution AFTER augmentation:")
for i, emotion in enumerate(EMOTIONS):
    count = np.sum(y_final == i)
    print(f"  {emotion}: {count} images")

# ─────────────────────────────────────────
# PLOT BEFORE VS AFTER
# ─────────────────────────────────────────
before_counts = [np.sum(y_train == i) for i in range(len(EMOTIONS))]
after_counts  = [np.sum(y_final == i) for i in range(len(EMOTIONS))]

x = np.arange(len(EMOTIONS))
width = 0.35

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, before_counts, width, label='Before', color='steelblue')
plt.bar(x + width/2, after_counts,  width, label='After',  color='coral')
plt.xticks(x, EMOTIONS)
plt.title('FER2013 — Class Distribution Before vs After Augmentation')
plt.xlabel('Emotion')
plt.ylabel('Number of Images')
plt.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/fer2013_augmentation_comparison.png')
plt.show()
print("Augmentation comparison plot saved.")

# ─────────────────────────────────────────
# SAVE AUGMENTED DATA
# ─────────────────────────────────────────
print("\nSaving augmented data...")
np.save(f'{SAVE_DIR}/X_train_augmented.npy', X_final)
np.save(f'{SAVE_DIR}/y_train_augmented.npy', y_final)
print("Saved successfully!")
print("\nFER2013 augmentation complete! ✅")