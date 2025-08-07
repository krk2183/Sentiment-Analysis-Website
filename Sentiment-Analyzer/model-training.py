import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data,test_data = datasets.load_dataset('imdb', split=["train","test"])

train_data.shape, test_data.shape

train_data.features, test_data.features

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def tokenize(example,tokenizer,max_len):
   tokens = tokenizer(example['text'])[:max_len]
   return {'tokens': tokens}

max_length = 256

train_data = train_data.map(
    tokenize, fn_kwargs={'tokenizer': tokenizer, "max_len": max_length}
)

test_data = test_data.map(
    tokenize, fn_kwargs={'tokenizer': tokenizer, "max_len": max_length}
)

tesize = 0.25

train_validation = train_data.train_test_split(test_size=tesize)

train_data = train_validation["train"]
valid_data = train_validation["test"]

len(train_data), len(valid_data), len(test_data)

"""# Creating a dictionary"""

min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

len(vocab)

vocab.get_itos()[:10]

vocab['and']

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']

'some_token' in vocab

vocab.set_default_index(unk_index)

vocab['some_token']

def numerilzie(example,vocab):
  ids = vocab.lookup_indices(example['tokens'])
  return {"id": ids}

train_data = train_data.map(
    numerilzie, fn_kwargs={'vocab': vocab}
)
valid_data =valid_data.map(
    numerilzie, fn_kwargs={'vocab': vocab}
)
test_data = test_data.map(
    numerilzie, fn_kwargs={'vocab': vocab}
)

train_data = train_data.with_format(type='torch', columns=['id','label'])
valid_data = valid_data.with_format(type='torch', columns=['id','label'])
test_data = test_data.with_format(type='torch', columns=['id','label'])

"""# *CREATING DATA LOADERS*
batch['ids'] should have shape of [batch_size, length]
** ------------------------------------------------------**
batch['label'] should have shape of [batch_size]
"""

def get_collate_fn(pad_index):
  def collate_fn(batch):
    batch_ids = [i['id'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(
        batch_ids, padding_value=pad_index, batch_first=True
    )
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'id': batch_ids, 'label': batch_label}
    return batch
  return collate_fn

def get_data_loader(dataset,batch_size, pad_index, shuffle=False):
  collate_fn = get_collate_fn(pad_index)
  data_loader=torch.utils.data.DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      collate_fn= collate_fn,
      shuffle=shuffle,
  )
  return data_loader

batch_size = 512

train_data_loader = get_data_loader(train_data,batch_size,pad_index,shuffle=True)
test_data_loader = get_data_loader(test_data,batch_size,pad_index)
valid_data_loader = get_data_loader(valid_data,batch_size,pad_index)

"""# Building the model"""

class NBow(nn.Module):
  def __init__(self,vocab_size,embedding_dim,output_dim,pad_index):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_index)
    self.fc = nn.Linear(embedding_dim,output_dim)

  def forward(self,ids):
    embedded = self.embedding(ids)
    pooled = embedded.mean(dim=1)
    prediction =self.fc(pooled)
    return prediction

vocab_size = len(vocab)
embedding_dim = 300
output_dim  = len(train_data.unique("label"))

model = NBow(vocab_size,embedding_dim,output_dim,pad_index)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Model has ', count_parameters(model),'trainable parameters')

vectors = torchtext.vocab.GloVe()
hello_vector = vectors.get_vecs_by_tokens('hello')

pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())

model.embedding.weight.data  = pretrained_embedding

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["id"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["id"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

n_epochs = 10
best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, criterion, optimizer, device
    )
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "nbow.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="train loss")
ax.plot(metrics["valid_losses"], label="valid loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()

model.load_state_dict(torch.load("nbow.pt"))

test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)

def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

namee = 'ready-model.pt'

torch.save(model.state_dict(),namee)

from collections import Counter
from nltk.lm import Vocabulary





display(train_data.unique("label"))

import datasets

train_data,test_data = datasets.load_dataset('imdb', split=["train","test"])

max_length = 256

train_data = train_data.map(
    tokenize, fn_kwargs={'tokenizer': tokenizer, "max_len": max_length}
)

test_data = test_data.map(
    tokenize, fn_kwargs={'tokenizer': tokenizer, "max_len": max_length}
)

tesize = 0.25

train_validation = train_data.train_test_split(test_size=tesize)

train_data = train_validation["train"]
valid_data = train_validation["test"]

min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']

vocab.set_default_index(unk_index)

def numerilzie(example,vocab):
  ids = vocab.lookup_indices(example['tokens'])
  return {"id": ids}

train_data = train_data.map(
    numerilzie, fn_kwargs={'vocab': vocab}
)
valid_data =valid_data.map(
    numerilzie, fn_kwargs={'vocab': vocab}
)
test_data = test_data.map(
    numerilzie, fn_kwargs={'vocab': vocab}
)

train_data = train_data.with_format(type='torch', columns=['id','label'])
valid_data = valid_data.with_format(type='torch', columns=['id','label'])
test_data = test_data.with_format(type='torch', columns=['id','label'])

display(train_data.unique("label"))

