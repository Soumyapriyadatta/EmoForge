import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
FACE_DIR    = 'face/data/processed'
SPEECH_DIR  = 'speech/data/processed'
TEXT_DIR    = 'text/data/processed'
SAVE_DIR    = 'utils/figures'
os.makedirs(SAVE_DIR, exist_ok=True)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")

# Face
X_face  = np.load(f'{FACE_DIR}/X_train_augmented.npy')
y_face  = np.load(f'{FACE_DIR}/y_train_augmented.npy')

# Speech
X_speech = np.load(f'{SPEECH_DIR}/X_train.npy')
y_speech = np.load(f'{SPEECH_DIR}/y_train.npy')

# Text
y_text = np.load(f'{TEXT_DIR}/train_labels.npy')

print("All data loaded!")

# ─────────────────────────────────────────
# FIGURE 1 — COMBINED CLASS DISTRIBUTION
# All 3 datasets side by side
# ─────────────────────────────────────────
face_counts   = [np.sum(y_face == i)   for i in range(7)]
speech_counts = [np.sum(y_speech == i) for i in range(7)]
text_counts   = [np.sum(y_text == i)   for i in range(7)]

x     = np.arange(len(EMOTIONS))
width = 0.25

plt.figure(figsize=(14, 6))
plt.bar(x - width,   face_counts,   width, label='FER2013 (Face)',      color='steelblue')
plt.bar(x,           speech_counts, width, label='RAVDESS (Speech)',     color='darkorange')
plt.bar(x + width,   text_counts,   width, label='GoEmotions (Text)',    color='seagreen')
plt.xticks(x, EMOTIONS)
plt.title('EmoForge — Class Distribution Across All Three Datasets')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emoforge_combined_distribution.png', dpi=300)
plt.show()
print("Figure 1 saved — Combined class distribution")

# ─────────────────────────────────────────
# FIGURE 2 — SAMPLE FACE IMAGES
# One sample per emotion
# ─────────────────────────────────────────
plt.figure(figsize=(14, 3))
for i, emotion in enumerate(EMOTIONS):
    idx = np.where(y_face == i)[0][0]
    plt.subplot(1, 7, i+1)
    plt.imshow(X_face[idx].reshape(48, 48), cmap='gray')
    plt.title(emotion, fontsize=9)
    plt.axis('off')
plt.suptitle('EmoForge — FER2013 Sample Images per Emotion', fontsize=12)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emoforge_face_samples.png', dpi=300)
plt.show()
print("Figure 2 saved — Face sample images")

# ─────────────────────────────────────────
# FIGURE 3 — SAMPLE MFCC HEATMAPS
# One per emotion
# ─────────────────────────────────────────
plt.figure(figsize=(14, 3))
for i, emotion in enumerate(EMOTIONS):
    idx = np.where(y_speech == i)[0][0]
    plt.subplot(1, 7, i+1)
    plt.imshow(X_speech[idx], aspect='auto', origin='lower', cmap='viridis')
    plt.title(emotion, fontsize=9)
    plt.axis('off')
plt.suptitle('EmoForge — RAVDESS Sample MFCCs per Emotion', fontsize=12)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emoforge_speech_samples.png', dpi=300)
plt.show()
print("Figure 3 saved — Speech MFCC samples")

# ─────────────────────────────────────────
# FIGURE 4 — DATASET SUMMARY TABLE
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
table_data = [
    ['Dataset', 'Modality', 'Train Samples', 'Test Samples', 'Classes'],
    ['FER2013',    'Face',   str(len(X_face)),    '7,178',  '7'],
    ['RAVDESS',    'Speech', str(len(X_speech)),  '288',    '7'],
    ['GoEmotions', 'Text',   str(len(y_text)),    '5,427',  '7'],
]
table = ax.table(cellText=table_data[1:],
                 colLabels=table_data[0],
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)
plt.title('EmoForge — Dataset Summary', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/emoforge_dataset_summary.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 4 saved — Dataset summary table")

print("\nEDA complete! ✅")
print(f"All figures saved to {SAVE_DIR}/")