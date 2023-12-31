# -*- coding: utf-8 -*-

!pip install transformers
!pip install datasets

from google.colab import drive
drive.mount('/content/drive')

import os
from transformers import BertForTokenClassification, BertTokenizerFast, BertModel
import torch
import keras
import torch.optim as optim
from collections import Counter
import tensorflow as tf
import torch.nn as nn

# English_train_file = '/content/drive/My Drive/A4/GED/English/en_fce_train.tsv'
# English_dev_file = '/content/drive/My Drive/A4/GED/English/en_fce_dev.tsv'
English_train_file = '/content/drive/My Drive/A4/GED/English/en_fce_train.tsv'
English_dev_file = '/content/drive/My Drive/A4/GED/English/en_fce_dev.tsv'
Czech_train_file = '/content/drive/My Drive/A4/GED/Czech/cs_geccc_train.tsv'
Czech_dev_file = '/content/drive/My Drive/A4/GED/Czech/cs_geccc_dev.tsv'
German_train_file = '/content/drive/My Drive/A4/GED/German/de_falko-merlin_train.tsv'
German_dev_file = '/content/drive/My Drive/A4/GED/German/de_falko-merlin_dev.tsv'
Italian_train_file = '/content/drive/My Drive/A4/GED/Italian/it_merlin_train.tsv'
Italian_dev_file = '/content/drive/My Drive/A4/GED/Italian/it_merlin_dev.tsv'
Swedish_train_file = '/content/drive/My Drive/A4/GED/Swedish/sv_swell_train.tsv'
Swedish_dev_file = '/content/drive/My Drive/A4/GED/Swedish/sv_swell_dev.tsv'

device_name = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

UNK = '[UNK]'
PAD = '[PAD]'

def get_vocabulary_and_data(data_file, max_vocab_size=None):
    vocab = Counter()
    pos_vocab = {'[CLS]','[SEP]'}
    vocab[UNK] = 1
    vocab[PAD] = 1
    data = []
    gold_labels = []
    with open(data_file, 'r', encoding='utf8') as f:
        # sent = ['[CLS]']
        # sent_pos = ['[CLS]']
        sent = []
        sent_pos = []
        for line in f:
            if line.strip():
                tok, pos = line.strip().split('\t')[0], line.strip().split('\t')[1]
                sent.append(tok)
                if pos == 'c':
                  pos = 0
                else:
                  pos = 1
                sent_pos.append(pos)
                vocab[tok]+=1
                vocab['[CLS'] += 1
                vocab['[SEP'] += 1
                pos_vocab.add(pos)
            elif sent:
                # sent.append('[SEP]')
                # sent_pos.append('[SEP]')
                data.append(sent)
                gold_labels.append(sent_pos)
                # sent = ['[CLS]']
                # sent_pos = ['[CLS]']
                sent = []
                sent_pos = []
                #sent_pos = []
    vocab = sorted(vocab.keys(), key = lambda k: vocab[k], reverse=True)
    if max_vocab_size:
        vocab = vocab[:max_vocab_size-2]
    vocab = [UNK, PAD] + vocab
    return {k:v for v,k in enumerate(vocab)}, list(pos_vocab), data, gold_labels

# vocab, labels, train_data, train_labels = get_vocabulary_and_data(English_train_file)
# vocab, labels, dev_data, dev_labels = get_vocabulary_and_data(English_dev_file)
English_vocab, English_labels, English_train_data, English_train_labels = get_vocabulary_and_data(English_train_file)
Czech_vocab, Czech_labels, Czech_train_data, Czech_train_labels = get_vocabulary_and_data(Czech_train_file)
German_vocab, German_labels, German_train_data, German_train_labels = get_vocabulary_and_data(German_train_file)
Italian_vocab, Italian_labels, Italian_train_data, Italian_train_labels = get_vocabulary_and_data(Italian_train_file)
Swedish_vocab, Swedish_labels, Swedish_train_data, Swedish_train_labels = get_vocabulary_and_data(Swedish_train_file)

train_data = English_train_data + Czech_train_data + German_train_data + Italian_train_data + Swedish_train_data
train_labels = English_train_labels + Czech_train_labels + German_train_labels + Italian_train_labels + Swedish_train_labels

_, _, English_dev_data, English_dev_labels = get_vocabulary_and_data(English_dev_file)
_, _, Czech_dev_data, Czech_dev_labels = get_vocabulary_and_data(Czech_dev_file)
_, _, German_dev_data, German_dev_labels = get_vocabulary_and_data(German_dev_file)
_, _, Italian_dev_data, Italian_dev_labels = get_vocabulary_and_data(Italian_dev_file)
_, _, Swedish_dev_data, Swedish_dev_labels = get_vocabulary_and_data(Swedish_dev_file)

dev_data = English_dev_data + Czech_dev_data + German_dev_data + Italian_dev_data + Swedish_dev_data
dev_labels = English_dev_labels + Czech_dev_labels + German_dev_labels + Italian_dev_labels + Swedish_dev_labels

train_input_str = []
for input in train_data:
  string = ' '.join(input)
  train_input_str.append(string)

dev_input_str = []
for input in dev_data:
  string = ' '.join(input)
  dev_input_str.append(string)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

train_tokenized = [tokenizer(token, padding='max_length', max_length=131, truncation=True, return_tensors="pt") for token in train_input_str]
dev_tokenized = [tokenizer(token, padding='max_length', max_length=131, truncation=True, return_tensors="pt") for token in dev_input_str]

# model = BertModel.from_pretrained("bert-base-multilingual-cased", num_labels=2)

test_data = train_tokenized[0]['input_ids']
test_mask = train_tokenized[0]['attention_mask']

hidden_dim = 128
dropout_rate = 0.5

# output = model(test_data, test_mask)
# lstm = nn.LSTM(input_size=model.config.hidden_size, hidden_size=hidden_dim, num_layers=2, dropout=dropout_rate, bidirectional=True)
# lstm_out, _ = lstm(output.last_hidden_state)
# lstm_last_hidden_state = lstm_out[:, -1]
# dropout = nn.Dropout(dropout_rate)
# dropout_out = dropout(lstm_last_hidden_state)
# linear = nn.Linear(hidden_dim * 2,131)
# logits = linear(dropout_out)
# test_label = torch.tensor(train_label_ids[0]).unsqueeze(0).float()
# loss_fn = torch.nn.BCEWithLogitsLoss()
# # print(logits)
# # print(test_label)
# loss = loss_fn(logits, test_label)
# # print(logits)

class BertLSTMModel(torch.nn.Module):
  def __init__(self):
    super(BertLSTMModel, self).__init__()
    self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
    self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_dim, num_layers=2, dropout=dropout_rate, bidirectional=True)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(hidden_dim * 2, 131)
  
  def forward(self, input_id, mask):
    output = self.bert(input_ids=input_id, attention_mask=mask)
    bert_hidden_state = output.last_hidden_state
    lstm_out, _ = self.lstm(bert_hidden_state)
    lstm_last_hidden_state = lstm_out[:, -1]
    dropout_out = self.dropout(lstm_last_hidden_state)
    logits = self.linear(dropout_out)
    return logits

model2 = BertLSTMModel()
model2.to(device_name)

def align_label_example(tokenized_input, labels):
  word_ids = tokenized_input.word_ids()
  previous_word_idx = None
  label_ids = []
  for word_idx in word_ids:
    if word_idx is None:
      label_ids.append(-100)
    elif word_idx != previous_word_idx:
      try:
        label_ids.append(labels[word_idx])
      except:
        label_ids.append(-100)
    else:
      label_ids.append(labels[word_idx] if False else -100)
    previous_word_idx = word_idx
  return label_ids

class DataSequence(torch.utils.data.Dataset):
  def __init__(self, input_data, input_label):
    self.texts = input_data
    self.labels = input_label

  def __len__(self):
    return len(self.labels)

  def get_batch_data(self, idx):
    return self.texts[idx]

  def get_batch_labels(self, idx):
    return torch.LongTensor(self.labels[idx])

  def __getitem__(self, idx):
    batch_data = self.get_batch_data(idx)
    batch_labels = self.get_batch_labels(idx)

    return batch_data, batch_labels

train_label_ids = []
for token, ls in zip(train_tokenized, train_labels):
  train_label_ids.append(align_label_example(token, ls))

dev_label_ids = []
for token, ls in zip(dev_tokenized, dev_labels):
  dev_label_ids.append(align_label_example(token, ls))

dev_length = len(dev_tokenized) // 2
test_length = len(dev_tokenized) - dev_length
print(dev_length, test_length)

train_dataset = DataSequence(train_tokenized, train_label_ids)
dev_dataset = DataSequence(dev_tokenized[:len(dev_tokenized)//2], dev_label_ids[:len(dev_tokenized)//2])
test_dataset = DataSequence(dev_tokenized[len(dev_tokenized)//2:], dev_label_ids[:len(dev_tokenized)//2:])

from torch.utils.data import DataLoader
from tqdm import tqdm

batch_size = 32
learning_rate = 2e-5
num_epochs = 2
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model2.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, num_workers=4, batch_size=32, shuffle=True)

# sigmoid = 0
# l = 0
# logits = 0
# loss = 0
# total_loss= 0
# total_acc = 0
# loss_fn = torch.nn.CrossEntropyLoss()
# for data, label in tqdm(train_dataloader):
#   ids = data['input_ids'].squeeze(1).to(device_name)
#   mask = data['attention_mask'].squeeze(1).to(device_name)
#   logits = model2(ids, mask)
#   # l = torch.Tensor(label).unsqueeze(0).float()
#   l = label.float().to(device_name)
#   loss = loss_fn(logits, l)
#   total_loss += loss.item()
#   # for i in range(logits.shape[0]):
#   #   logits_clean = logits[i][label[i] != -100]
#   #   print(logits_clean)
#   #   label_clean = label[i][label[i] != -100]

#   #   predictions = logits_clean.argmax(-1)
#   #   print(predictions)
#   #   acc = (predictions == label_clean).float().mean()
#   #   total_acc += acc
#   #   break
#   # print(total_acc)

#   sigmoid = torch.sigmoid(logits.view(-1))
#   preds = (sigmoid > 0.5).float()
#   print(preds)
#   num_correct = (preds == l).sum().item()
#   print(num_correct)
  break

# print(l)
# preds = (sigmoid > 0.5).squeeze(-1).float()
# print(preds)
# total_acc = 0
# print(logits.shape[0])
# for i in tqdm(range(len(logits))):
#   logits_clean = logits[i][l[i] != -100]
#   label_clean = l[i][label[i] != -100]

#   predictions = logits_clean.argmax(dim=-1)
#   # print(label_clean)
#   # print(predictions)
#   acc = (predictions == label_clean).float().mean()
#   total_acc += acc
# print(total_acc)
# # print(sigmoid)
# # print(preds)
# print(loss)

from sklearn.metrics import f1_score

for epoch in range(num_epochs):
  model2.train()
 
  for data, label in tqdm(train_dataloader):
    total_acc_train = 0
    total_loss_train = 0

    id = data['input_ids'].squeeze(1).to(device_name)
    mask = data['attention_mask'].squeeze(1).to(device_name)
    label = label.float().to(device_name)

    optimizer.zero_grad()
    logits = model2(id, mask)

    loss = loss_fn(logits, label)

    for i in range(logits.shape[0]):
      logits_clean = logits[i][label[i] != -100]
      label_clean = label[i][label[i] != -100]

      predictions = logits_clean.argmax(dim=-1)
      acc = (predictions == label_clean).float().mean()
      total_acc_train += acc
      total_loss_train += loss.item()

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

  model2.eval()
  total_acc_dev = 0
  total_loss_dev = 0
  total_F1_score = 0

  for data, label in dev_dataloader:
    label = label.float().to(device_name)
    mask = data['attention_mask'].squeeze(1).to(device_name)
    id = data['input_ids'].squeeze(1).to(device_name)
    logits = model2(id, mask)
    loss = loss_fn(logits, label)

    for i in range(logits.shape[0]):
      logits_clean = logits[i][label[i] != -100]
      label_clean = label[i][label[i] != -100]
      predictions = logits_clean.argmax(dim=-1)
      acc = (predictions == label_clean).float().mean()
      total_acc_dev += acc
      total_loss_dev += loss.item()

  #print(f'Epochs:{epoch + 1} | Train Loss: {total_loss_train} | Train Accuracy: {total_acc_train} | Dev Loss: {total_loss_dev} | Dev Accuracy: {total_acc_dev}')
  print(f'Epochs: {epoch + 1}| Accuracy: {total_acc_dev / dev_length: .3f}')

def evaluation(model, test_dataset):

  test_dataLoader = DataLoader(test_dataset, batch_size=32, shuffle=True)
  total_test_acc = 0
  for data, label in test_dataLoader:
    label = label.float().to(device_name)
    mask = data['attention_mask'].squeeze(1).to(device_name)
    id = data['input_ids'].squeeze(1).to(device_name)

    logits = model(id, mask)

    loss = loss_fn(logits, label)

    for i in range(logits.shape[0]):
      logits_clean = logits[i][label[i] != -100]
      label_clean = label[i][label[i] != -100]

      predictions = logits_clean.argmax(dim=-1)
      acc = (predictions == label_clean).float().mean()
      total_test_acc += acc
  print(f'Test Accuracy: {total_test_acc / test_length: .3f}')

evaluation(model2, test_dataset)
