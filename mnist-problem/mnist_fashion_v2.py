import tensorflow as tf


def mnist_classifier():
    mnist = tf.keras.datasets.mnist

    # NORMALIZE YOUR IMAGE HERE
    (train_image, train_label), (test_image, test_label) = mnist.load_data()

    # train_image = train_image.astype('float32')
    # test_image = test_image.astype('float32')

    # train_image = tf.cast(train_image, tf.float32)
    # test_image = tf.cast(test_image, tf.float32)

    train_image = train_image/255
    test_image = test_image/255

    imageTrainShape = train_image.shape
    print(imageTrainShape)

    imageTestShape = test_image.shape
    print(imageTestShape)

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.summary()

    # COMPILE MODEL HERE
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    # TRAIN YOUR MODEL HERE
    model.fit(train_image,
              train_label,
              validation_data=(test_image,test_label),
              epochs=100)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = mnist_classifier()
    model.save("model_C2.h5")
