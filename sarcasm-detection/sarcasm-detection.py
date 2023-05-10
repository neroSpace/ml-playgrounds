import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sarcasm_detection():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open("./sarcasm.json", "r") as f:
        dataset = json.load(f)

    for data in dataset:
        sentences.append(data['headline'])
        labels.append(data['is_sarcastic'])

    # Split the sentences
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]

    # Split the labels
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Fit tokenizer with training data
    tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)

    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_sentences = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_sentences = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # label_token = Tokenizer()
    # label_token.fit_on_texts(training_labels)

    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    model.fit(training_sentences, training_labels, validation_data=(testing_sentences, testing_labels), epochs=50, verbose=1)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = sarcasm_detection()
    model.save("sarcasm_detection.h5")
