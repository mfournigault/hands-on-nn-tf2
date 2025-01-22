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

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
    
# Exercice 3
# Define a training class with the following methods:
# - constructor: takes the model as input and initializes the loss, optimizer, step, and metrics
# - train_step: takes the input and labels as input, computes the loss, gradients, and updates the model
# - train: takes the training data as input and trains the model for a given number of epochs
class ClassificationTrainer:
    def __init__(self, model):
        self.model = model
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.step = tf.Variable(1, name="global_step")
        self.optimizer = tf.optimizers.Adam(1e-3)
        self.accuracy = tf.metrics.Accuracy()
        self.mean_loss = tf.metrics.Mean(name='loss')
        self.step = tf.Variable(1, name="global_step")
        self.ckpt = tf.train.Checkpoint(step=self.step, optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)
        self.train_summary_writer = tf.summary.create_file_writer('./log/train')
    
    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            logits = self.model(inputs)
            loss_value = self.loss(labels, logits)
        
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.step.assign_add(1)
        
        self.accuracy.update_state(labels, tf.argmax(logits, -1))
        return loss_value, self.accuracy.result()
    
    # @tf.function : no use of autograph in this part, because it would require to write in a 
    # complete different manner the loop and use of checkpoints, ...
    def train(self, train_x, train_y, epochs=10, batch_size=32):
        tf.print(tf.executing_eagerly())
        self.manager.latest_checkpoint
        self.ckpt.restore(self.manager.latest_checkpoint)
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            tf.print(f"Restored from {self.manager.latest_checkpoint}")
        else:
            tf.print("Initializing from scratch.")
        nr_batches_train = int(train_x.shape[0] / batch_size)
        tf.print(f"Batch size: {batch_size}")
        tf.print(f"Number of batches per epoch: {nr_batches_train}")

        with self.train_summary_writer.as_default():
            for epoch in range(epochs):
                for t in range(nr_batches_train):
                    start_from = t * batch_size
                    to = (t + 1) * batch_size
                    features, labels = train_x[start_from:to], train_y[start_from:to]

                    loss_value, accuracy_value = self.train_step(features, labels)
                    self.mean_loss.update_state(loss_value)

                    if t % 10 == 0:
                        tf.print(
                            f"{self.step.numpy()}: {loss_value} - accuracy: {accuracy_value}"
                        )
                        save_path = self.manager.save()
                        tf.print(f"Checkpoint saved: {save_path}")
                        tf.summary.image('train_set', features, max_outputs=3, step=self.step.numpy())
                        tf.summary.scalar('accuracy', accuracy_value, step=self.step.numpy())
                        tf.summary.scalar('loss', self.mean_loss.result(), step=self.step.numpy())
                        self.accuracy.reset_state()
                        self.mean_loss.reset_state()
                tf.print(f"Epoch {epoch} terminated")
                # Measuring accuracy on the whole training set at the end of epoch
                for t in range(nr_batches_train):
                    start_from = t * batch_size
                    to = (t + 1) * batch_size
                    features, labels = train_x[start_from:to], train_y[start_from:
                                                                    to]
                    logits = self.model(features)
                    self.accuracy.update_state(labels, tf.argmax(logits, -1))
                tf.print(f"Training accuracy: {self.accuracy.result()}")
                self.accuracy.reset_state()

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
