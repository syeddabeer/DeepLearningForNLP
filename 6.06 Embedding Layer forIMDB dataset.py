from keras.dataset import imdb
# imdb dataset is retrieved as integers in keras.dataset

from keras import preprocessing

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_text, maxlen=maxlen) 


from keras.models import Sequential
from keras.models import Dense, Flatten

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history=model.fit(x_train, y_train,
	epochs=10,
	batch_size=32,
	validation_split=0.2)

	