import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def make_model(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes)
    ])


def load_data():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # Scale input in [-1, 1] range
    train_x = tf.expand_dims(train_x, -1)
    train_x = (tf.image.convert_image_dtype(train_x, tf.float32) - 0.5) * 2
    train_y = tf.expand_dims(train_y, -1)

    test_x = test_x / 255. * 2 - 1
    test_x = (tf.image.convert_image_dtype(test_x, tf.float32) - 0.5) * 2
    test_y = tf.expand_dims(test_y, -1)

    return (train_x, train_y), (test_x, test_y)

# Exercise 1a
# Define with functional API the model previously defined as sequential
def make_model_func(intpus, n_classes):
    # inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, (5, 5), activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(n_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Exercise 1b
# Define with subclassing the model previously defined as sequential
class FminstClassifier(tf.keras.Model):
    def __init__(self, n_classes):
        super(FminstClassifier, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32, (5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1))
        self.maxpool1 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)
        self.maxpool2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
    
# Exercice 3
# Define a training class with the following methods using keras training loops
class ClassificationTrainer:
    def __init__(self, model):
        self.model = model
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.step = tf.Variable(1, name="global_step")
        self.optimizer = tf.optimizers.Adam(1e-3)
        self.accuracy = tf.metrics.Accuracy()
        self.mean_loss = tf.metrics.Mean(name='loss')
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
    
    # @tf.function : no use of autograph in this part, because it would require to write in a 
    # complete different manner the loop and use of checkpoints, ...
    def train(self, train_x, train_y, epochs=10, batch_size=32):
        tf.print(tf.executing_eagerly())
        # metrics and losses are traced at the frequency update_freq
        # update_freq can be "batch" or "epoch", if "batch" is chosen, the metrics are traced at each batch
        # this can be customised by overwriting the method train_step
        logdir = "log/keras/"
        tb_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
        checkpoints_filepath = "keras_ckpts/checkpoint.model.keras"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_filepath, 
            # save_weights_only=True,
            monitor = "val_accuracy",
            mode = "auto",
            save_freq='epoch')
        training_history = self.model.fit(train_x, train_y, 
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[cp_callback, tb_callback],
            validation_split=0.2,
            verbose=1 # progress bar
            )
        
        print("Average training accuracy: ", training_history.history['val_accuracy'])
        print("Average training loss: ", training_history.history['loss'])

        

def train():
    # Define the model
    n_classes = 10
    model = FminstClassifier(n_classes)

    # Input data
    (train_x, train_y), (test_x, test_y) = load_data()

    trainer = ClassificationTrainer(model)
    trainer.train(train_x, train_y, epochs=10, batch_size=32)

if __name__ == "__main__":
    train()
