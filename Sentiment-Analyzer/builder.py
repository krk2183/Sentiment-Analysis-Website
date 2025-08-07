import torch
import torch.nn as nn
import os
import collections
import json
import random
import re

# Import the datasets library for loading IMDb data
from datasets import load_dataset
# DataLoader is a core PyTorch utility and is still used for efficient batching.
from torch.utils.data import DataLoader

# --- 1. NBOW MODEL DEFINITION (Consistent across all files) ---
# This model architecture must be consistent across all files.
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

# Simple tokenizer function (consistent with app.py)
def my_tokenizer(text):
    text = text.lower()
    # Remove HTML tags (common in IMDb reviews)
    text = re.sub(r'<br />', ' ', text)
    # Keep only lowercase letters, numbers, spaces, and apostrophes
    text = re.sub(r"[^a-z0-9 ']", "", text)
    tokens = text.split()
    return tokens

# --- 3. VOCABULARY BUILDING (Manual, adapted for datasets library) ---

def build_vocab_manual(data, min_freq=1, specials=None):
    """
    Manually builds a vocabulary from a Hugging Face Dataset object.
    It iterates through the 'text' field of each item in the dataset.
    """
    if specials is None:
        specials = []
    
    word_counts = collections.Counter()
    # Iterate over items in the dataset and access the 'text' field
    for item in data: # <--- CHANGED: Iterate over 'item' (dictionary)
        word_counts.update(my_tokenizer(item['text'])) # <--- CHANGED: Access item['text']
    
    # Assign IDs: special tokens first, then words by frequency (or alphabetically if same freq)
    vocab = {token: i for i, token in enumerate(specials)}
    current_index = len(specials)
    
    # Sort words by frequency (descending) then alphabetically (ascending)
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

    for word, count in sorted_words:
        if count >= min_freq and word not in vocab:
            vocab[word] = current_index
            current_index += 1
            
    return vocab

# --- 4. DATASET LOADING AND VOCABULARY CREATION (using datasets library) ---

print("Loading IMDb dataset using Hugging Face 'datasets' library...")
# Load the IMDb dataset. This will automatically download it if not cached.
# It returns a DatasetDict, from which we access the 'train' and 'test' splits.
try:
    train_dataset = load_dataset('imdb', split='train') # <--- Using datasets.load_dataset
    test_dataset = load_dataset('imdb', split='test')   # <--- Using datasets.load_dataset
    print(f"Loaded {len(train_dataset)} training reviews and {len(test_dataset)} test reviews.")
except Exception as e:
    print(f"Error loading IMDb dataset: {e}")
    print("Please ensure you have an active internet connection to download the dataset.")
    exit()

print("Building vocabulary...")
# Build vocabulary from training data using the manual function, passing the dataset object
vocab = build_vocab_manual(train_dataset, min_freq=1, specials=["<unk>", "<pad>"]) # min_freq=1 for larger vocab
# Ensure <unk> and <pad> indices are correctly set
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

# Save vocabulary to vocab.json
vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
with open(vocab_path, "w") as f:
    json.dump(vocab, f, indent=4) 
print(f"Vocabulary saved to {vocab_path}")

# --- 5. DATA PREPARATION FOR TRAINING (Collate Function for DataLoader) ---

# MAX_SEQ_LEN is the maximum sequence length for input texts.
# Reviews longer than this will be truncated, shorter ones will be padded.
MAX_SEQ_LEN = 256 # Increased for better context capture and potentially higher accuracy.

def collate_batch(batch):
    """
    Collates a batch of data (list of dictionaries from datasets library) into PyTorch tensors.
    This function is used by the DataLoader to process each batch.
    """
    label_list, text_list = [], []
    for item in batch: # <--- CHANGED: Iterate over 'item' (dictionary)
        label_list.append(item['label']) # <--- CHANGED: Access item['label']
        
        # Tokenize the text and convert tokens to their numerical IDs using the vocabulary.
        processed_text = torch.tensor([vocab.get(token, UNK_INDEX) for token in my_tokenizer(item['text'])], dtype=torch.long) # <--- CHANGED: Access item['text']
        
        # Apply padding or truncation to ensure all sequences in a batch have MAX_SEQ_LEN.
        if len(processed_text) < MAX_SEQ_LEN:
            padded_text = torch.cat([processed_text, torch.full((MAX_SEQ_LEN - len(processed_text),), PAD_INDEX, dtype=torch.long)])
        else:
            padded_text = processed_text[:MAX_SEQ_LEN]
            
        text_list.append(padded_text)
    
    # Stack the processed texts and labels into single tensors for the model.
    # labels are unsqueezed to have a shape of [batch_size, 1] which is expected by BCELoss.
    return torch.stack(text_list), torch.tensor(label_list, dtype=torch.float).unsqueeze(1)

# --- 6. TRAINING CONFIGURATION ---
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100 
OUTPUT_DIM = 1      

# Determine the device (GPU if available, otherwise CPU) for training.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate the NBow model and move it to the selected device.
model = NBow(VOCAB_SIZE, EMBEDDING_DIM, OUTPUT_DIM, PAD_INDEX).to(DEVICE)

# Define the Loss Function (Binary Cross-Entropy for binary classification)
criterion = nn.BCELoss() 
# Define the Optimizer (Adam is a popular choice for deep learning).
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 20 # Number of times the model will iterate over the entire training dataset. Increased for thorough training.
BATCH_SIZE = 64 # Number of samples processed in each training step.

# Create PyTorch DataLoaders for efficient batching and shuffling.
# train_dataset and test_dataset are now Hugging Face Dataset objects.
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# --- 7. TRAINING LOOP ---
print("\nStarting model training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set the model to training mode.
    total_loss = 0
    
    # Iterate over batches provided by the train_dataloader.
    for batch_ids, batch_labels in train_dataloader:
        # Move batch tensors to the appropriate device (CPU/GPU).
        batch_ids, batch_labels = batch_ids.to(DEVICE), batch_labels.to(DEVICE)

        optimizer.zero_grad() # Clear gradients from the previous optimization step.
        
        predictions = model(batch_ids) # Perform a forward pass to get predictions.
        
        loss = criterion(predictions, batch_labels) # Calculate the loss.
        loss.backward() # Perform a backward pass to compute gradients.
        optimizer.step() # Update model weights based on the computed gradients.

        total_loss += loss.item() # Accumulate the loss for reporting.

    avg_loss = total_loss / len(train_dataloader) # Calculate average loss for the epoch.
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {avg_loss:.4f}")

    # --- 8. EVALUATION ---
    model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates, etc.).
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad(): # Disable gradient calculation during evaluation for efficiency.
        # Iterate over batches provided by the test_dataloader.
        for batch_ids, batch_labels in test_dataloader:
            batch_ids, batch_labels = batch_ids.to(DEVICE), batch_labels.to(DEVICE)
            
            predictions = model(batch_ids)
            # Convert probabilities to binary predictions (0 or 1) based on a 0.5 threshold.
            predicted_labels = (predictions > 0.5).float() 
            
            # Count correct predictions and total samples.
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
    
    accuracy = correct_predictions / total_samples # Calculate accuracy for the epoch.
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Test Accuracy: {accuracy:.4f}")

print("\nTraining complete!")

# --- 9. SAVE THE TRAINED MODEL ---
# Define the path to save the trained model's state dictionary.
model_path = os.path.join(os.path.dirname(__file__), "new-model.pt")
torch.save(model.state_dict(), model_path)
print(f"Trained model saved to {model_path}")
