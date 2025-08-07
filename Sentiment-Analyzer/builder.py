import torch
import torch.nn as nn
import os
import collections
import json
import random
import re

# We will be using IMDB dataset which is a compilation of movie reviews
from datasets import load_dataset
from torch.utils.data import DataLoader

# --- 1. NBOW MODEL CREATION ---
class NBow(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids):
        embedded = self.embedding(ids)
        pooled = embedded.mean(dim=1)
        prediction = self.fc(pooled)
        return self.sigmoid(prediction)

# --- 2. TOKENIZER ---

# Simple tokenizer (torchtext can also be used in here for simplicity, but in the recent 
# versions of PyTorch current versions of torchtext aren't supported and will result in an error during installation)
def my_tokenizer(text):
    text = text.lower()
    # Remove HTML tags (common in IMDb reviews)
    text = re.sub(r'<br />', ' ', text)
    # Keep only lowercase letters, numbers, spaces, and apostrophes
    text = re.sub(r"[^a-z0-9 ']", "", text)
    tokens = text.split()
    return tokens

# --- 3. BUILDING THE VOCABULARY ---

def build_vocab_manual(data, min_freq=1, specials=None):
    """
    Manually builds a vocabulary from a Hugging Face Dataset object.
    It iterates through the 'text' field of each item in the dataset.
    """
    if specials is None:
        specials = []
    
    word_counts = collections.Counter()
    for item in data: 
        word_counts.update(my_tokenizer(item['text'])) 
    
    vocab = {token: i for i, token in enumerate(specials)}
    current_index = len(specials)
    # Words are first sorted by frequency and then alphabetically
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

    for word, count in sorted_words:
        if count >= min_freq and word not in vocab:
            vocab[word] = current_index
            current_index += 1
            
    return vocab

# --- 4. DATASET LOADING AND VOCABULARY CREATION (using datasets library) ---

print("Loading IMDb dataset using Hugging Face library...")
# Returns a DatasetDict 
# Handles connection issues
try:
    train_dataset = load_dataset('imdb', split='train') 
    test_dataset = load_dataset('imdb', split='test')  
    print(f"Loaded {len(train_dataset)} training reviews and {len(test_dataset)} test reviews.")
except Exception as e:
    print(f"Error loading IMDb dataset: {e}")
    print("Please ensure you have an active internet connection to download the dataset.")
    exit()

print("Building vocabulary...")
vocab = build_vocab_manual(train_dataset, min_freq=1, specials=["<unk>", "<pad>"]) # min_freq=1 for larger vocab
if "<unk>" in vocab:
    UNK_INDEX = vocab["<unk>"]
else:
    UNK_INDEX = len(vocab) 
    vocab["<unk>"] = UNK_INDEX
    
if "<pad>" in vocab:
    PAD_INDEX = vocab["<pad>"]
else:
    PAD_INDEX = len(vocab)
    vocab["<pad>"] = PAD_INDEX

print(f"Vocabulary size: {len(vocab)}")

vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
with open(vocab_path, "w") as f:
    json.dump(vocab, f, indent=4) 
print(f"Vocabulary saved to {vocab_path}")

# --- 5. DATA PREPARATION FOR TRAINING (Collate Function for DataLoader) ---

# MAX_SEQ_LEN is the max sequence length
# Reviews longer will be shortened, shorter ones will be extended via padding
MAX_SEQ_LEN = 256 

def collate_batch(batch):
    """
    Collates a batch of data (list of dictionaries from datasets library) into PyTorch tensors.
    This function is used by the DataLoader to process each batch.
    """
    label_list, text_list = [], []
    for item in batch: 
        label_list.append(item['label']) 
        
        processed_text = torch.tensor([vocab.get(token, UNK_INDEX) for token in my_tokenizer(item['text'])], dtype=torch.long) # <--- CHANGED: Access item['text']
        
        if len(processed_text) < MAX_SEQ_LEN:
            padded_text = torch.cat([processed_text, torch.full((MAX_SEQ_LEN - len(processed_text),), PAD_INDEX, dtype=torch.long)])
        else:
            padded_text = processed_text[:MAX_SEQ_LEN]
            
        text_list.append(padded_text)
    
    return torch.stack(text_list), torch.tensor(label_list, dtype=torch.float).unsqueeze(1)

# --- 6. TRAINING CONFIGURATION ---
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100 
OUTPUT_DIM = 1      

# Checks if a GPU is available if not it will default to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NBow(VOCAB_SIZE, EMBEDDING_DIM, OUTPUT_DIM, PAD_INDEX).to(DEVICE)

# Binary Cross-Entropy chosen since binary classification
criterion = nn.BCELoss() 
# Adam optimizer is suitable for this use case
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 20 
BATCH_SIZE = 64 

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# --- 7. TRAINING LOOP ----
print("\nStarting model training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Enable training mode
    total_loss = 0
    
    for batch_ids, batch_labels in train_dataloader:
        batch_ids, batch_labels = batch_ids.to(DEVICE), batch_labels.to(DEVICE)

        optimizer.zero_grad() 
        
        predictions = model(batch_ids) 
        
        loss = criterion(predictions, batch_labels) 
        loss.backward() 
        optimizer.step() 

        total_loss += loss.item() 

    avg_loss = total_loss / len(train_dataloader) # Since Avg Loss will be outputed this line calculates that
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {avg_loss:.4f}")

    # --- 8. EVALUATION ---
    model.eval() # Evaluation mode disables dropout and batch norm updates which is what we need for our use case
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad(): 
        for batch_ids, batch_labels in test_dataloader:
            batch_ids, batch_labels = batch_ids.to(DEVICE), batch_labels.to(DEVICE)
            
            predictions = model(batch_ids)
            # Probability to binary (threshold being 0.5)
            predicted_labels = (predictions > 0.5).float() 
            
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
    
    accuracy = correct_predictions / total_samples 
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Test Accuracy: {accuracy:.4f}")

print("\nTraining complete!")

# --- 9. SAVE THE MODEL ---
model_path = os.path.join(os.path.dirname(__file__), "new-model.pt")

torch.save(model.state_dict(), model_path)

print(f"Trained model saved to {model_path}")
