# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import os
import tensorflow as tf

from collections import Counter

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from keras.activations import softmax
from keras.optimizers import Adadelta
from keras.optimizers import Adam

import numpy as np
from keras import backend as K

!pip install transformers

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
UNK = '[UNK]'
PAD = '[PAD]'

def get_vocabulary_and_data(data_file, max_vocab_size=None):
    vocab = Counter()
    #pos_vocab = {'<s>','</s>'}
    pos_vocab = {'i', 'c'}
    vocab[UNK] = 1
    vocab[PAD] = 1
    data = []
    gold_labels = []
    with open(data_file, 'r', encoding='utf8') as f:
        # sent = ['<s>']
        # sent_pos = ['<s>']
        sent = []
        sent_pos = []
        for line in f:
            if line.strip():
                tok, pos = line.strip().split('\t')[0], line.strip().split('\t')[1]
                sent.append(tok)
                sent_pos.append(pos)
                vocab[tok]+=1
                # vocab['<s>'] += 1
                # vocab['</s>'] += 1
                pos_vocab.add(pos)
            elif sent:
                # sent.append('</s>')
                # sent_pos.append('</s>')
                data.append(sent)
                gold_labels.append(sent_pos)
                # sent = ['<s>']
                # sent_pos = ['<s>']
                sent = []
                sent_pos = []
    vocab = sorted(vocab.keys(), key = lambda k: vocab[k], reverse=True)
    if max_vocab_size:
        vocab = vocab[:max_vocab_size-2]
    vocab = [UNK, PAD] + vocab
    return {k:v for v,k in enumerate(vocab)}, list(pos_vocab), data, gold_labels


def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]


def one_hot_encode_label(label, label_set):
    vec = [1.0 if l==label else 0.0 for l in label_set]
    return np.array(vec)

def clean(seqs, vocab, unk):
    for i,seq in enumerate(seqs):
        for j,tok in enumerate(seq):
            if tok>=len(vocab):
                seq[j] = unk

def batch_generator(data, labels, vocab, label_set, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for sent, sent_pos in zip(data,labels):
            batch_x.append(vectorize_sequence(sent, vocab))
            batch_y.append([one_hot_encode_label(label, label_set) for label in sent_pos])
            if len(batch_x) >= batch_size:
                clean(batch_x, vocab, vocab[UNK])
                # Pad Sequences in batch to same length
                batch_x = pad_sequences(batch_x, vocab[PAD])
                batch_y = pad_sequences(batch_y, one_hot_encode_label(PAD, label_set))
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []


def describe_data(data, gold_labels, label_set, generator):
    batch_x, batch_y = [], []
    for bx, by in generator:
        batch_x = bx
        batch_y = by
        break
    print('Data example:',data[0])
    print('Label:',gold_labels[0])
    print('Label count:', len(label_set))
    print('Data size', len(data))
    print('Batch input shape:', batch_x.shape)
    print('Batch output shape:', batch_y.shape)


def pad_sequences(batch_x, pad_value):
    ''' This function should take a batch of sequences of different lengths
        and pad them with the pad_value token so that they are all the same length.

        Assume that batch_x is a list of lists.
    '''
    pad_length = len(max(batch_x, key=lambda x: len(x)))
    for i, x in enumerate(batch_x):
        if len(x) < pad_length:
            batch_x[i] = x + ([pad_value] * (pad_length - len(x)))

    return batch_x

epochs = 5 # number of epochs
learning_rate = 2e-5 # learning rate
dropout = 0.3 # dropout rate
early_stopping = -1 # early stopping criteria
embedding_size = 100 # embedding dimension size
hidden_size = 10 # hidden layer size
batch_size = 64 # batch size

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  device_name = '/cpu:0'
  print(
      '\n\n This notebook is not '
      'configured to use a GPU.  You can change this in Notebook Settings. Defaulting to:' + device_name)
else:
  print ('GPU Device found: ' + device_name)

English_vocab, English_labels, English_train_data, English_train_labels = get_vocabulary_and_data(English_train_file)
Czech_vocab, Czech_labels, Czech_train_data, Czech_train_labels = get_vocabulary_and_data(Czech_train_file)
German_vocab, German_labels, German_train_data, German_train_labels = get_vocabulary_and_data(German_train_file)
Italian_vocab, Italian_labels, Italian_train_data, Italian_train_labels = get_vocabulary_and_data(Italian_train_file)
Swedish_vocab, Swedish_labels, Swedish_train_data, Swedish_train_labels = get_vocabulary_and_data(Swedish_train_file)
vocab = Counter()
vocab.update(English_vocab)
vocab.update(Czech_vocab)
vocab.update(German_vocab)
vocab.update(Italian_vocab)
vocab.update(Swedish_vocab)
labels = English_labels
train_data = English_train_data + Czech_train_data + German_train_data + Italian_train_data + Swedish_train_data
train_labels = English_train_labels + Czech_train_labels + German_train_labels + Italian_train_labels + Swedish_train_labels

#_, _, dev_data, dev_labels = get_vocabulary_and_data(dev_file)
_, _, English_dev_data, English_dev_labels = get_vocabulary_and_data(English_dev_file)
_, _, Czech_dev_data, Czech_dev_labels = get_vocabulary_and_data(Czech_dev_file)
_, _, German_dev_data, German_dev_labels = get_vocabulary_and_data(German_dev_file)
_, _, Italian_dev_data, Italian_dev_labels = get_vocabulary_and_data(Italian_dev_file)
_, _, Swedish_dev_data, Swedish_dev_labels = get_vocabulary_and_data(Swedish_dev_file)

dev_data = English_dev_data + Czech_dev_data + German_dev_data + Italian_dev_data + Swedish_dev_data
dev_labels = English_dev_labels + Czech_dev_labels + German_dev_labels + Italian_dev_labels + Swedish_dev_labels

from transformers import BertModel
from tensorflow.keras.layers import Lambda

bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
bert_embedding_layer = bert_model.embeddings

#_, _, test_data, test_labels = get_vocabulary_and_data(test_file)

# describe_data(train_data, train_labels, labels,
#               batch_generator(train_data, train_labels, vocab, labels, batch_size))

with tf.device(device_name):

    # Implement your model here! ----------------------------------------------------------------------
    # Use the variables batch_size, hidden_size, embedding_size, dropout, epochs
    embedding_layer = Embedding(len(vocab), embedding_size)
    LSTM_layer = LSTM(hidden_size, dropout=dropout, return_sequences=True)
    bidirectional_LSTM_layer = Bidirectional(LSTM(hidden_size, return_sequences=True))
    dense_layer = Dense(len(labels), activation='softmax')
    dropout_layer = Dropout(dropout)
    #conv1d_layer = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    #max_pooling_layer = GlobalMaxPooling1D()
    dropout_layer2 = SpatialDropout1D(0.2)
    GED_model = Sequential([embedding_layer, bidirectional_LSTM_layer, dropout_layer, TimeDistributed(dense_layer), dropout_layer])
    #GED_model = Sequential([embedding_layer, dropout_layer, bidirectional_LSTM_layer, dropout_layer, conv1d_layer, TimeDistributed(dense_layer),  dropout_layer])
    # ------------------------------------------------------------------------------------------------

    GED_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    #pos_tagger.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    best_acc = 0

    for i in range(epochs):
        print('Epoch',i+1,'/',epochs)
        # Training
        GED_model.fit(batch_generator(train_data, train_labels, vocab, labels, batch_size),
                                  epochs=1, steps_per_epoch=len(train_data)/batch_size)
        # Evaluation
        loss, acc = GED_model.evaluate(batch_generator(dev_data, dev_labels, vocab, labels),
                                                  steps=len(dev_data))
        print('Dev Loss:', loss, 'Dev Acc:', acc)

    # test_lose, test_acc = best_model.evaluate(batch_generator(test_data, test_labels, vocab, labels), steps=len(test_data))
    # print('Test Loss:', test_lose, 'Test_Acc:', test_acc)

describe_data(train_data, train_labels, labels,
              batch_generator(train_data, train_labels, vocab, labels, batch_size))

# input_text = "This a sample sentence"
# input_tokens = ['<s>'] + input_text.split() + ['</s>']
# input_ids = pad_sequences(input_tokens, vocab[PAD])
sample = 'I sf i fer gegerg s .'
vectorized_sample = [vectorize_sequence(sample.split(), vocab)]
padded_sample = pad_sequences(vectorized_sample, vocab[PAD])
predictions = GED_model.predict(padded_sample)
print(predictions)
predictions = [labels[np.argmax(prediction)] for prediction in predictions[0]]
print(predictions)

print(train_data[0], train_labels[0], labels)
