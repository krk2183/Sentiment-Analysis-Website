# --- 1. IMPORTS  ---
import re, os, torch
from flask_cors import CORS
import torch.nn as nn
from flask import Flask, request, jsonify, send_file 
import json 

# Change these lines in case you have renamed the files
vocab_file_name = 'vocab.json'
model_file_name = 'new-model.pt'

# Some tunable parameters
threshold = 0.5
PAD_TO = 50

# INITIALIZE THE APP
app = Flask(__name__)
CORS(app) # Cross-Origin Resource Sharing aka. CORS

# --- 2. NBOW MODEL DEFINITION ---
class NBow(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.sigmoid = nn.Sigmoid() # sigmoid can be used for binary classification

    def forward(self, ids):
        embedded = self.embedding(ids) # this line embeds batch_size, seq_len and embedding_dim
        pooled = embedded.mean(dim=1) # this one pools: batch_size and embedding_dim 
        prediction = self.fc(pooled) # the last prediction layer. Returns the previous parameters and layers' outputs
        return self.sigmoid(prediction)

# --- 3. HELPER FUNCTIONS AND GLOBAL VARIABLES ---

# As said torchtext is problematic so I have defined a custom tokenizer that is simple and just as quick if not quicker
def my_tokenizer(text):
    text = text.lower()
    # Only keep lowercase letters alongside numbers that have spaces and apostrophes
    text = re.sub(r"[^a-z0-9 ']", "", text)
    tokens = text.split()
    return tokens

# Load the vocab.json file 
def load_vocab():
    vocab_path = os.path.join(os.path.dirname(__file__), vocab_file_name)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}. Please ensure it exists.")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab

# The global variables that the Flask routes will use, loaded once at app startup
try:
    VOCAB = load_vocab()
    VOCAB_SIZE = len(VOCAB)
    PAD_INDEX = VOCAB["<pad>"] # Get the padding index from the loaded vocabulary
    EMBEDDING_DIM = 100 # Must match the dimension used when creating the model in build_sentiment_model.py
    OUTPUT_DIM = 1      # Must match the dimension used when creating the model in build_sentiment_model.py
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Determine if CUDA (GPU) is available

    # Initialize the model with the correct parameters
    MODEL = NBow(VOCAB_SIZE, EMBEDDING_DIM, OUTPUT_DIM, PAD_INDEX).to(DEVICE)
    
    # Load the model state dictionary from 'new-model.pt'
    model_path = os.path.join(os.path.dirname(__file__), model_file_name)
    if not os.path.exists(model_path):
        # This warning should ideally not be hit if build_sentiment_model.py is run first
        print(f"Warning: {model_path} not found. Creating a dummy model file.")
        dummy_model = NBow(VOCAB_SIZE, EMBEDDING_DIM, OUTPUT_DIM, PAD_INDEX)
        torch.save(dummy_model.state_dict(), model_path)
        print("Dummy model created. Please replace with your trained model.")

    MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    MODEL.eval()  # Set the model to evaluation mode (important for inference)
    print("Sentiment model and vocabulary loaded successfully.")

except Exception as e:
    print(f"Initialization Error: {e}")
    print("Please ensure you have created 'new-model.pt' and 'vocab.json' in the same directory by running 'build_sentiment_model.py'.")
    exit() # Exit the app if there's a critical loading error

# --- 4. FLASK ROUTES ---

# Route to serve the main HTML file (index.html)
@app.route('/')
def serve_index():
    index_html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    print(f"Attempting to serve index.html from: {index_html_path}") # Debugging print
    if not os.path.exists(index_html_path):
        return "Error: index.html not found in the same directory as app.py", 404
    return send_file(index_html_path)

# API endpoint for sentiment prediction
@app.route("/predict_sentiment", methods=['POST'])
def predict_sentiment_route():
    try:
        data = request.get_json()
        text_input = data["text"]
        
        # Tokenize the input text
        tokens = my_tokenizer(text_input)
        
        # Convert tokens to numerical IDs
        # Use .get with default for robustness against words not in vocab
        ids = [VOCAB.get(token, VOCAB.get('<unk>', 0)) for token in tokens]
        
        # Padding: Pad the sequence of IDs to a fixed length
        # This is CRUCIAL for nn.Embedding which expects fixed-size inputs.
        # Choose a reasonable MAX_SEQ_LEN based on your expected input text length.
        # This value MUST be consistent between training (if you had it) and inference.
        MAX_SEQ_LEN = PAD_TO # Example: pad to 50 tokens. Adjust as needed for longer sentences.
        if len(ids) < MAX_SEQ_LEN:
            ids.extend([PAD_INDEX] * (MAX_SEQ_LEN - len(ids)))
        else:
            ids = ids[:MAX_SEQ_LEN] # Truncate if too long

        input_tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(DEVICE) 

        # Make the prediction
        with torch.no_grad(): 
            prediction = MODEL(input_tensor).squeeze(dim=0)
        
        prediction_value = prediction.item()
        
        # The result is declared in here with threshold 0.5 (in the hyperparameters section)
        sentiment = "Positive" if prediction_value > threshold else "Negative"

        return jsonify({
            'sentiment_score': prediction_value,
            'sentiment': sentiment
        })
    except Exception as e:
        # Print the full traceback for easier debugging
        import traceback
        traceback.print_exc() 
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
