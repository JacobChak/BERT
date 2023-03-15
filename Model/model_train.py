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

# 读取文件并进行转换
train_df, test_df = get_train_test_pd()
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=256)
print('begin encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]

train_df['x'] = train_df['text'].apply(f)
test_df['x'] = test_df['text'].apply(f)
print('end encoding')

# 训练集和测试集
x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])
# print('x_train: ', x_train.shape)

# 将类型y值转化为ont-hot向量
num_classes = 4
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 模型结构：BERT + 双向GRU + Attention + FC
inputs = Input(shape=(256, 768, ))
gru = Bidirectional(GRU(256, dropout=0.2, return_sequences=True))(inputs)
attention = Attention(32)(gru)
output = Dense(num_classes, activation='softmax')(attention)
model = Model(inputs, output)

# 模型可视化
# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

# 如果原来models文件夹下存在.h5文件，则全部删除
model_dir = '/content/drive/MyDrive/People_2/people_relation_extract/models'
if os.listdir(model_dir):
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))

# 保存最新的val_acc最好的模型文件
filepath="/content/drive/MyDrive/People_2/people_relation_extract/models/per-rel-{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

# 模型训练以及评估
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=30, callbacks=[early_stopping, checkpoint])
# model.save('people_relation.h5')

print('在测试集上的效果：', model.evaluate(x_test, y_test))

# 读取关系对应表
with open('/content/drive/MyDrive/People_2/people_relation_extract/data/rel_dict.json', 'r', encoding='utf-8') as f:
    label_id_dict = json.loads(f.read())

sorted_label_id_dict = sorted(label_id_dict.items(), key=itemgetter(1))
values = [_[0] for _ in sorted_label_id_dict]

# 输出每一类的classification report
y_pred = model.predict(x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=values))

# 绘制loss和acc图像
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
plt.savefig("/content/drive/MyDrive/People_2/people_relation_extract/loss_acc.png")
plt.savefig('/content/drive/MyDrive/People_2/people_relation_extract/loss_acc.eps',dpi=800,format = 'eps')

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
plt.savefig("/content/drive/MyDrive/People_2/people_relation_extract/ROC_AUC.png")
plt.savefig('/content/drive/MyDrive/People_2/people_relation_extract/ROC_AUC.eps',dpi=800,format = 'eps')
