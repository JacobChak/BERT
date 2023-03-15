# -*- coding: utf-8 -*-
# 模型训练
import os
os.environ['PATH'] += ':/usr/local/cuda-10.0/bin'
os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from att import Attention
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from operator import itemgetter

from load_data import get_train_test_pd
from bert.extract_feature import BertVector

import math

# Set the maximum sequence length for BERT
max_seq_len = 512

# Define a function to encode text using BERT
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=max_seq_len)

def encode_text(text):
    # Split the text into smaller chunks of length <= max_seq_len
    chunks = []
    for i in range(0, len(text), max_seq_len):
        chunk = text[i:i+max_seq_len]
        chunks.append(chunk)

    # Encode each chunk using BERT
    encoded_chunks = []
    for chunk in chunks:
        encoded_chunk = bert_model.encode([chunk])["encodes"][0]
        encoded_chunks.append(encoded_chunk)

    # Concatenate the encoded chunks
    if len(encoded_chunks) == 1:
        encoded_text = encoded_chunks[0]
    else:
        encoded_text = np.concatenate(encoded_chunks, axis=0)

    return encoded_text

# Encode the training and test data
train_df, test_df = get_train_test_pd()
train_df['x'] = train_df['text'].apply(encode_text)
test_df['x'] = test_df['text'].apply(encode_text)

# Prepare the training and test data for the model
x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])

num_classes = 4
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the model architecture
inputs = Input(shape=(None, 768))
gru = Bidirectional(GRU(256, dropout=0.2, return_sequences=True))(inputs)
attention = Attention(32)(gru)
output = Dense(num_classes, activation='softmax')(attention)
model = Model(inputs, output)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Set the batch size for training
batch_size = 16

# Train the model
num_train_samples = len(train_df)
num_test_samples = len(test_df)
num_train_steps = math.ceil(num_train_samples/batch_size)
num_test_steps = math.ceil(num_test_samples/batch_size)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
filepath="/content/drive/MyDrive/People_2/per-rel-{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    batch_size=batch_size, epochs=30,
                    steps_per_epoch=num_train_steps, validation_steps=num_test_steps,
                    callbacks=[early_stopping, checkpoint])

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Read the relationship correspondence table
with open('/content/drive/MyDrive/People_2/data/rel_dict.json', 'r', encoding='utf-8') as f:
    label_id_dict = json.loads(f.read())

sorted_label_id_dict = sorted(label_id_dict.items(), key=itemgetter(1))
values = [_[0] for _ in sorted_label_id_dict]

# classification report
y_pred = model.predict(x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=values))

# Draw the loss and acc print
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig("/content/drive/MyDrive/People_2/loss_acc.png")
plt.savefig('/content/drive/MyDrive/People_2/loss_acc.eps',dpi=800,format = 'eps')

# Print the ROC curve and AUC

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
roc_auc = auc(fpr, tpr)
print("AUC : ", roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("/content/drive/MyDrive/People_2/ROC_AUC.png")
plt.savefig('/content/drive/MyDrive/People_2/ROC_AUC.eps',dpi=800,format = 'eps')
