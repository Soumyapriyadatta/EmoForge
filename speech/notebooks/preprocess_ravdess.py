import numpy as np
import librosa
import os
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RAVDESS_DIR = 'speech/data/Audio_Speech_Actors_01-24'
SAVE_DIR    = 'speech/data/processed'
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────
# RAVDESS EMOTION MAPPING
# Filename format: 03-01-XX-01-01-01-01.wav
# 3rd number is emotion code
# ─────────────────────────────────────────
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'neutral',   # calm → neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

EMOTIONS      = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# ─────────────────────────────────────────
# EXTRACT MFCC FEATURES
# ─────────────────────────────────────────
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        # Pad or truncate to fixed length
        if mfcc.shape[1] < max_len:
            pad = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ─────────────────────────────────────────
# LOAD ALL RAVDESS FILES
# ─────────────────────────────────────────
features = []
labels   = []
skipped  = 0

print("Loading RAVDESS audio files...")

for actor_folder in sorted(os.listdir(RAVDESS_DIR)):
    actor_path = os.path.join(RAVDESS_DIR, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    if actor_folder == 'processed':
        continue
    for file_name in os.listdir(actor_path):
        if not file_name.endswith('.wav'):
            continue
        parts = file_name.split('-')
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            skipped += 1
            continue
        file_path = os.path.join(actor_path, file_name)
        mfcc = extract_mfcc(file_path)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(EMOTION_TO_IDX[emotion])

print(f"Loaded {len(features)} audio files")
print(f"Skipped {skipped} files")

# ─────────────────────────────────────────
# CONVERT TO NUMPY
# ─────────────────────────────────────────
X = np.array(features)
y = np.array(labels)

print(f"\nFeature shape: {X.shape}")
print(f"Labels shape:  {y.shape}")

# ─────────────────────────────────────────
# CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\nClass distribution:")
for i, emotion in enumerate(EMOTIONS):
    count = np.sum(y == i)
    print(f"  {emotion}: {count} samples")

# ─────────────────────────────────────────
# PLOT CLASS DISTRIBUTION
# ─────────────────────────────────────────
plt.figure(figsize=(10, 5))
counts = [np.sum(y == i) for i in range(len(EMOTIONS))]
plt.bar(EMOTIONS, counts, color='darkorange')
plt.title('RAVDESS — Class Distribution')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/ravdess_class_distribution.png')
plt.show()
print("Class distribution plot saved.")

# ─────────────────────────────────────────
# PLOT SAMPLE MFCC
# ─────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.imshow(X[0], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.title(f'Sample MFCC — {EMOTIONS[y[0]]}')
plt.xlabel('Time Frames')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/ravdess_sample_mfcc.png')
plt.show()
print("Sample MFCC plot saved.")

# ─────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set:  {X_test.shape}")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
print("\nSaving processed data...")
np.save(f'{SAVE_DIR}/X_train.npy', X_train)
np.save(f'{SAVE_DIR}/X_test.npy',  X_test)
np.save(f'{SAVE_DIR}/y_train.npy', y_train)
np.save(f'{SAVE_DIR}/y_test.npy',  y_test)
print("Saved successfully!")
print("\nRAVDESS preprocessing complete! ✅")