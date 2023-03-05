import tensorflow as tf
import numpy as np
import pandas as pd
import os

dictionary = pd.read_csv("training_data/product_names.csv")["Product_Name"].to_list()
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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(len(dictionary),)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(dictionary), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        # train model
    x_train = preprocess(dictionary, 3)
    y_train = tf.keras.utils.to_categorical(range(len(dictionary)))
    model.fit(x_train, y_train, epochs=69, batch_size=2)
    model.save("spell_checker_model")
    
    
        
    # else:
    #     new_layers = tf.keras.models.Sequential([
    #             tf.keras.layers.Dense(128, activation='relu', input_shape=(len(dictionary),)),
    #             tf.keras.layers.Dense(64, activation='relu'),
    #             tf.keras.layers.Dense(len(dictionary), activation='softmax')
    #         ], "new_layers")
    #     pretrained_model = tf.keras.models.load_model("spell_checker_model")
    #     model = tf.keras.models.Sequential([pretrained_model, new_layers])
    #     model.compile(optimizer='adam',
    #                     loss='categorical_crossentropy',
    #                     metrics=['accuracy'])
    #     for layer in pretrained_model.layers:
    #         layer.trainable = False
    #     x_train = preprocess(name, 3)
    #     y_train = tf.keras.utils.to_categorical(range(len(dictionary)))
    #     model.fit(x_train, y_train, epochs=80, batch_size=2)
    #     model.save("spell_checker_model")

# define function to predict closest word in dictionary


def predict_closest_word(word):
    # load saved model
    model = tf.keras.models.load_model("spell_checker_model")
    x = preprocess([word], 3)
    y_pred = model.predict(x)
    closest_word_idx = np.argmax(y_pred)
    closest_word_prob = y_pred[0, closest_word_idx]
    closest_word = dictionary[closest_word_idx]
    return closest_word, closest_word_prob

# just call this pls
def get_closest_word(word):
    if not os.path.exists("spell_checker_model"):
        start_model_processing()
    closest_word, confidence = get_closest_word(word)
    # print(closest_word)
    # print(confidence)
    return closest_word


def add_new_recall_product(name):
    global dictionary
    
    if os.path.exists("spell_checker_model"):
        return
        
    every_product = pd.read_csv("training_data/product_names.csv")["Product_Name"]
    if name not in every_product.to_list():
        every_product.loc[len(every_product)] = name
        every_product.to_csv("training_data/product_names.csv", index=False)
    
    
    recalled_product = pd.read_csv("training_data/recalled.csv")["Product_Name"]
    if name not in recalled_product.to_list():
        recalled_product.loc[len(recalled_product)] = name
        recalled_product.to_csv("training_data/recalled.csv", index=False)
    
    start_model_processing(name)
    
    return True


def remove_recall_product(name):
    recalled = pd.read_csv("training_data/recalled.csv")["Product_Name"]
    recalled.drop(recalled[recalled == name].index, inplace=True)
    recalled.to_csv("training_data/recalled.csv", index=False)
    return True


# start_model_processing()
# print(predict_closest_word("sammi shaanxi cold noodle"))