# -*- coding: utf-8 -*-

!pip install transformers
!pip install datasets

from google.colab import drive
drive.mount('/content/drive')

import os
from transformers import BertForTokenClassification, BertTokenizerFast
import torch
from datasets import load_dataset
import pandas as pd
from keras.activations import softmax
from keras.optimizers import Adam
import keras
import torch.optim as optim
from collections import Counter
import tensorflow as tf

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

print(train_data[:3])
print(train_labels[:3])

train_input_str = []
for input in train_data:
  string = ' '.join(input)
  train_input_str.append(string)

dev_input_str = []
for input in dev_data:
  string = ' '.join(input)
  dev_input_str.append(string)

model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
#model.cuda()
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

train_tokenized = [tokenizer(token, padding='max_length', max_length=131, truncation=True, return_tensors="pt") for token in train_input_str]
dev_tokenized = [tokenizer(token, padding='max_length', max_length=131, truncation=True, return_tensors="pt") for token in dev_input_str]

print(train_tokenized[0])
print(train_labels[0])
print(dev_tokenized[0])
print(dev_labels[0])

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

train_label_ids = []
for token, ls in zip(train_tokenized, train_labels):
  train_label_ids.append(align_label_example(token, ls))

dev_label_ids = []
for token, ls in zip(dev_tokenized, dev_labels):
  dev_label_ids.append(align_label_example(token, ls))

print(train_label_ids[1])
print(train_labels[1])
print(dev_label_ids[1])
print(dev_labels[1])

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

class BertModel(torch.nn.Module):
  def __init__(self):
    super(BertModel, self).__init__()
    self.bert = model
  
  def forward(self, input_id, mask, label):
    output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
    return output

dev_length = len(dev_tokenized) // 2
test_length = len(dev_tokenized) - dev_length

train_dataset = DataSequence(train_tokenized, train_label_ids)
dev_dataset = DataSequence(dev_tokenized[:len(dev_tokenized)//2], dev_label_ids[:len(dev_tokenized)//2])
test_dataset = DataSequence(dev_tokenized[len(dev_tokenized)//2:], dev_label_ids[:len(dev_tokenized)//2:])

from torch.utils.data import DataLoader
from tqdm import tqdm

batch_size = 32
learning_rate = 2e-5
num_epochs = 2
loss_fn = torch.nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, num_workers=4, batch_size=32, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.to(device_name)

from sklearn.metrics import f1_score

model = BertModel()
best_acc = 0
best_loss = 1000
for epoch in range(num_epochs):
  model.train()
  for data, label in tqdm(train_dataloader):
    total_acc_train = 0
    total_loss_train = 0
    label = label.to(device_name)
    mask = data['attention_mask'].squeeze(1).to(device_name)
    id = data['input_ids'].squeeze(1).to(device_name)

    optimizer.zero_grad()
    loss,logits = model(id, mask, label)

    for i in range(logits.shape[0]):
      logits_clean = logits[i][label[i] != -100]
      label_clean = label[i][label[i] != -100]

      predictions = logits_clean.argmax(dim=1)
      acc = (predictions == label_clean).float().mean()
      total_acc_train += acc
      total_loss_train += loss.item()

    loss.backward()
    optimizer.step()
    
  model.eval()

  total_acc_dev = 0
  total_loss_dev = 0
  total_F1_score = 0

  for data, label in dev_dataloader:
    label = label.to(device_name)
    mask = data['attention_mask'].squeeze(1).to(device_name)
    id = data['input_ids'].squeeze(1).to(device_name)
    loss, logits = model(id, mask, label)

    for i in range(logits.shape[0]):
      logits_clean = logits[i][label[i] != -100]
      label_clean = label[i][label[i] != -100]
      predictions = logits_clean.argmax(dim=1)
      acc = (predictions == label_clean).float().mean()
      total_acc_dev += acc
      total_loss_dev += loss.item()

      f1 = f1_score(label_clean.cpu(), predictions.cpu(), average='weighted')
      total_F1_score += f1
  #print(f'Epochs:{epoch + 1} | Train Loss: {total_loss_train} | Train Accuracy: {total_acc_train} | Dev Loss: {total_loss_dev} | Dev Accuracy: {total_acc_dev}')
  # print(f'Epochs: {epoch + 1}| Val_Loss: {total_loss_dev / len(dev_data): .3f} | Accuracy: {total_acc_dev / (len(dev_data)/2): .3f} | F1: {total_F1_score / (len(dev_data)/2): .3f}')
   print(f'Epochs: {epoch + 1}| Val_Loss: {total_loss_dev / dev_length: .3f} | Accuracy: {total_acc_dev / dev_length: .3f} | F1: {total_F1_score / dev_length: .3f}')

def evaluation(model, test_dataset):

  test_dataLoader = DataLoader(test_dataset, num_workers=4, batch_size=32, shuffle=True)
  total_test_acc = 0
  for data, label in test_dataLoader:
    label = label.to(device_name)
    mask = data['attention_mask'].squeeze(1).to(device_name)
    id = data['input_ids'].squeeze(1).to(device_name)

    loss, logits = model(id, mask, label)

    for i in range(logits.shape[0]):
      logits_clean = logits[i][label[i] != -100]
      label_clean = label[i][label[i] != -100]

      predictions = logits_clean.argmax(dim=1)
      acc = (predictions == label_clean).float().mean()
      total_test_acc += acc
  print(f'Test Accuracy: {total_test_acc / test_length: .3f} | F1 Score: {total_F1_score / test_length: .3f}')

evaluation(model, test_dataset)

def align_word_ids(texts):
  
    token = tokenizer(texts, padding='max_length', max_length=131, truncation=True)
    word_ids = token.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if False else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

def generate(model, sentence):
  token = tokenizer(sentence, padding='max_length', max_length=131, truncation=True, return_tensors="pt")
  token_ids = token['input_ids'].to(device_name)
  masks = token['attention_mask'].to(device_name)
  label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device_name)

  logits = model(token_ids, masks, None)
  logits_clean = logits[0][label_ids != -100]

  predictions = logits_clean.argmax(dim=1).tolist()
  prediction_label = [i for i in predictions]
  print(sentence)
  print(prediction_label)

model_parameters_file = '/content/drive/My Drive/A4/GED/BERT_GED_model_weight_2.pt'
torch.save(model.state_dict(), model_parameters_file)
model_entire_file = '/content/drive/My Drive/A4/GED/BERT_GED_model_2.pt'
torch.save(model, model_entire_file)
