import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
DATA_DIR = 'text/data'
SAVE_DIR = 'text/data/processed'
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────
# LOAD TSV FILES
# ─────────────────────────────────────────
print("Loading GoEmotions files...")
train_df = pd.read_csv(f'{DATA_DIR}/train.tsv', sep='\t', header=None)
dev_df   = pd.read_csv(f'{DATA_DIR}/dev.tsv',   sep='\t', header=None)
test_df  = pd.read_csv(f'{DATA_DIR}/test.tsv',  sep='\t', header=None)

train_df.columns = ['text', 'labels', 'id']
dev_df.columns   = ['text', 'labels', 'id']
test_df.columns  = ['text', 'labels', 'id']

print(f"Train: {len(train_df)} samples")
print(f"Dev:   {len(dev_df)} samples")
print(f"Test:  {len(test_df)} samples")

# ─────────────────────────────────────────
# GOEMOTIONS 27 → 7 EMOTION MAPPING
# ─────────────────────────────────────────
EMOTION_MAP = {
    '0':  'happy',    # admiration
    '1':  'happy',    # amusement
    '2':  'angry',    # anger
    '3':  'angry',    # annoyance
    '4':  'happy',    # approval
    '5':  'disgust',  # caring → neutral (closest)
    '6':  'surprise', # confusion
    '7':  'happy',    # curiosity → happy
    '8':  'sad',      # desire → sad
    '9':  'sad',      # disappointment
    '10': 'disgust',  # disapproval
    '11': 'disgust',  # disgust
    '12': 'sad',      # embarrassment
    '13': 'happy',    # excitement
    '14': 'fear',     # fear
    '15': 'happy',    # gratitude
    '16': 'sad',      # grief
    '17': 'happy',    # joy
    '18': 'happy',    # love
    '19': 'neutral',  # nervousness → neutral
    '20': 'happy',    # optimism
    '21': 'angry',    # pride → angry
    '22': 'happy',    # realization → happy
    '23': 'sad',      # relief → sad
    '24': 'sad',      # remorse
    '25': 'sad',      # sadness
    '26': 'surprise', # surprise
    '27': 'neutral',  # neutral
}

EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# ─────────────────────────────────────────
# MAP LABELS
# ─────────────────────────────────────────
def map_label(label_str):
    # Each sample can have multiple labels — take first one
    first_label = str(label_str).split(',')[0].strip()
    emotion = EMOTION_MAP.get(first_label, 'neutral')
    return EMOTION_TO_IDX[emotion]

print("\nMapping labels...")
train_df['mapped_label'] = train_df['labels'].apply(map_label)
dev_df['mapped_label']   = dev_df['labels'].apply(map_label)
test_df['mapped_label']  = test_df['labels'].apply(map_label)

# ─────────────────────────────────────────
# CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\nClass distribution (training set):")
for i, emotion in enumerate(EMOTIONS):
    count = (train_df['mapped_label'] == i).sum()
    print(f"  {emotion}: {count} samples")

# ─────────────────────────────────────────
# PLOT CLASS DISTRIBUTION
# ─────────────────────────────────────────
plt.figure(figsize=(10, 5))
counts = [(train_df['mapped_label'] == i).sum() for i in range(len(EMOTIONS))]
plt.bar(EMOTIONS, counts, color='seagreen')
plt.title('GoEmotions — Training Set Class Distribution (Mapped to 7)')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/goemotions_class_distribution.png')
plt.show()
print("Class distribution plot saved.")

# ─────────────────────────────────────────
# TOKENIZE WITH DISTILBERT
# ─────────────────────────────────────────
print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(texts, max_length=128):
    return tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='np'
    )

print("Tokenizing train set...")
train_encodings = tokenize(train_df['text'])
print("Tokenizing dev set...")
dev_encodings   = tokenize(dev_df['text'])
print("Tokenizing test set...")
test_encodings  = tokenize(test_df['text'])

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
print("\nSaving processed data...")
np.save(f'{SAVE_DIR}/train_input_ids.npy',      train_encodings['input_ids'])
np.save(f'{SAVE_DIR}/train_attention_mask.npy', train_encodings['attention_mask'])
np.save(f'{SAVE_DIR}/train_labels.npy',         train_df['mapped_label'].values)

np.save(f'{SAVE_DIR}/dev_input_ids.npy',        dev_encodings['input_ids'])
np.save(f'{SAVE_DIR}/dev_attention_mask.npy',   dev_encodings['attention_mask'])
np.save(f'{SAVE_DIR}/dev_labels.npy',           dev_df['mapped_label'].values)

np.save(f'{SAVE_DIR}/test_input_ids.npy',       test_encodings['input_ids'])
np.save(f'{SAVE_DIR}/test_attention_mask.npy',  test_encodings['attention_mask'])
np.save(f'{SAVE_DIR}/test_labels.npy',          test_df['mapped_label'].values)

print("Saved successfully!")
print("\nGoEmotions preprocessing complete! ✅")