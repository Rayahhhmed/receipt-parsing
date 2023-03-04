import tensorflow as tf
import numpy as np
import pandas as pd


# dictionary = ["cat", "dog", "saiya is", "fish", "chicken", "giraffe", "appul"]

# define function to extract character n-grams from a word
def extract_ngrams(word, n):
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i:i+n])
    return ngrams

# define function to preprocess data
def preprocess(data, n):
    x = []
    for word in data:
        ngrams = extract_ngrams(word, n)
        features = np.zeros(len(dictionary))
        for ngram in ngrams:
            for i, dict_word in enumerate(dictionary):
                if ngram in extract_ngrams(dict_word, n):
                    features[i] += 1
        x.append(features)
    return np.array(x)





# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(dictionary),)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(dictionary), activation='softmax')
])

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model
x_train = preprocess(dictionary, 3)
y_train = tf.keras.utils.to_categorical(range(len(dictionary)))
model.fit(x_train, y_train, epochs=50, batch_size=2)

# define function to predict closest word in dictionary
def predict_closest_word(word):
    x = preprocess([word], 3)
    y_pred = model.predict(x)
    return dictionary[np.argmax(y_pred)]

# example usage
misspelled_word = "chikn"
closest_word = predict_closest_word(misspelled_word)
print(f"Closest word to '{misspelled_word}' is '{closest_word}'")