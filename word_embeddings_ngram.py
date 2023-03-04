import tensorflow as tf
import numpy as np
import pandas as pd
import os

dictionary = pd.read_csv("training_data/product_names.csv")["Product_Name"].to_list()
print(len(dictionary))
def extract_ngrams(word, n):
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i: i + n])
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



def start_model_processing():
    # define model
    with tf.device('/device:GPU:0'):
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
        model.fit(x_train, y_train, epochs=200, batch_size=2)

        # save model to disk
        model.save("spell_checker_model")

# define function to predict closest word in dictionary
def predict_closest_word(word):
    with tf.device('/device:GPU:0'):
        # load saved model
        model = tf.keras.models.load_model("spell_checker_model")
        x = preprocess([word], 3)
        y_pred = model.predict(x)
        return dictionary[np.argmax(y_pred)]

# just call this pls
def get_closest_word(word):
    if not os.path.exists("spell_checker_model"):
        start_model_processing()
    closest_word = predict_closest_word(word)
    return closest_word

