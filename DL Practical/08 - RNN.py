import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

num_words = 10000
maxlen = 200

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=num_words)

x_train=pad_sequences(x_train,maxlen=maxlen)
x_test=pad_sequences(x_test,maxlen=maxlen)

model=Sequential([
    Embedding(num_words,32,input_length=maxlen),
    SimpleRNN(64),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

print("Accuracy:",model.evaluate(x_test,y_test)[1])