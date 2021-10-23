# deployment of DL spam classification
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model, Model,  model_from_json
from keras.preprocessing import sequence
from keras import backend as K
from keras.utils.vis_utils import plot_model
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
import nltk.data
import keras
import pydot
import graphviz
print(keras.__version__)
import os.path as osp
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
import numpy as np
np.set_printoptions(threshold=np.inf)
import os

MAX_FEATURES = 10000
MAX_SENTENCE_LENGTH = 220

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Read training data and generate vocabulary
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
DATA_DIR='/Users/zuni/Documents/hbku/Research/Keras with LSTM/smsspamcollection'
ftrain = open(os.path.join(DATA_DIR, "SMSSpamCollection.txt"), 'rb')
for line in ftrain:
    label, sentence = line.strip().split(b'\t')
    words = nltk.word_tokenize(sentence.decode("ascii", "ignore").lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1
ftrain.close()

print(maxlen)
print(len(word_freqs))

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in
              enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

#writing the dictionary to a text file for use in android application
fo = open('file.txt', "w")
for k, v in index2word.items():
    fo.write(str(v) + ':'+ str(k) + '\n')
fo.close()

# convert sentences to sequences
X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
ftrain = open(os.path.join(DATA_DIR, "SMSSpamCollection.txt"), 'rb')
for line in ftrain:
    label, sentence = line.strip().split(b'\t')
    #Converting the ham and spam to 1 and 0 respecively
    if label == b'ham':
        label = 1
    else:
        label = 0
    words = nltk.word_tokenize(sentence.decode("ascii", "ignore").lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i] = seqs
    y[i] = int(label)
    i += 1
ftrain.close()

# Pad the sequences (left padded with zeros)
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
y = np.reshape(y,(len(y),1),order='F')


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

#build model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH, name="I"))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid',name='output_activation_node'))
model.summary()

# model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['mae','accuracy'])

history =  model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))
model.save('first_try.h5')

#evaluation
score, x, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, %.3f, accuracy: %.3f" % (score, x, acc))


y_pred = model.predict_proba(Xtest)

fpr = dict()
tpr = dict()
roc_auc = dict()
plt.figure()
lw = 2
fpr, tpr, _ = roc_curve(ytest, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

matthews_corrcoef(ytest.round(), y_pred.round())

for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,220)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print("pred %.f label %d %s" % (ypred, ylabel, sent))

model.get_layer('I')
# print [gd.node.value for node in model]

#trying to build a .pb file

input_fld = '/Users/zuni/Documents/hbku/Research/Keras with LSTM'
weight_file = 'first_try.h5'
num_output = 1
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
output_graph_name = 'constant_graph_weights.pb'

output_fld = input_fld + '/tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)
net_model = load_model(weight_file_path)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))



#
#
#
#
#

with tf.gfile.FastGFile("/Users/zuni/Documents/hbku/Research/Keras with LSTM/tensorflow_model/constant_graph_weights.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph_def.summary
graph_def.layers[0].input
model.layers[0].input
model.layers[0].output
model.layers[1].input
model.layers[1].output
model.layers[2].input
model.layers[2].output


m = get_simple_model()
check_model(m,mat_texts_tr,y_train,mat_texts_tst,y_test)