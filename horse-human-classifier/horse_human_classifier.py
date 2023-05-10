import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def horse_human_classifier():
    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    TRAINING_DIR = 'data/horse-or-human/'
    train_datagen = ImageDataGenerator(
        rescale=1./255)

    VALIDATION_DIR = 'data/validation-horse-or-human/'
    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator=train_datagen.flow_from_directory(TRAINING_DIR,
                                                      target_size=(150, 150),
                                                      batch_size=128,
                                                      class_mode="binary")

    valid_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                             target_size=(150, 150),
                                                             batch_size=128,
                                                             class_mode="binary")

    class myCallback(tf.keras.callbacks.Callback):
        # Define the correct function signature for on_epoch_end
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.83 and logs.get('val_accuracy')>0.83:  # @KEEP
                print("\nReached 83% accuracy & val_accuracy so cancelling training!")

                # Stop training once the above condition is met
                self.model.stop_training = True

    callbacks=myCallback()

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy']
                  )

    model.fit(
        train_generator,
        validation_data=valid_generator,
        # steps_per_epoch=8,
        epochs=10,
        # callbacks=[callbacks],
        verbose=1

    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = horse_human_classifier()
    model.save("horse_human_classifier.h5")


