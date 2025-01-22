import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np


def prepare_datasets(batch_size=32):
    """
    Prepares and filters the VOC datasets for object detection and classification.

    This function loads the VOC dataset using TensorFlow Datasets (tfds) and splits it into
    training, testing, and validation sets. It then filters each dataset to include only
    images with a single object annotated.

    Returns:
        tuple: A tuple containing the filtered training, testing, and validation datasets.
    """
    # Train, test, and validation are datasets for object detection: multiple objects per image.
    (train, test, validation), info = tfds.load(
        "voc", split=["train", "test", "validation"], with_info=True
    )

    print(info)

    # Create a subset of the dataset by filtering the elements: we are interested
    # in creating a dataset for object detetion and classification that is a dataset
    # of images with a single object annotated.
    def filter(dataset):
        return dataset.filter(lambda row: tf.equal(tf.shape(row["objects"]["label"])[0], 1))


    train, test, validation = filter(train), filter(test), filter(validation)

    def preprocess(dataset):
        def _fn(row):
            row["image"] = tf.image.convert_image_dtype(row["image"], tf.float32)
            row["image"] = tf.image.resize(row["image"], (299, 299))
            return row

        return dataset.map(_fn)

    train, test, validation = preprocess(train), preprocess(test), preprocess(validation)
    train, test, validation = train.cache(), test.cache(), validation.cache()
    train, test, validation = train.batch(batch_size), test.batch(batch_size), validation.batch(batch_size)
    train, test, validation = train.prefetch(tf.data.AUTOTUNE), test.prefetch(tf.data.AUTOTUNE), validation.prefetch(tf.data.AUTOTUNE)

    return train, test, validation


class MObjectDetector(tf.keras.Model):

    def __init__(self, n_classes):
        super(MObjectDetector, self).__init__()
        # Input layer
        self._inputs = tf.keras.layers.Input(shape=(299, 299, 3))

        # Feature extractor
        self._feature_extractor = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
            output_shape=[2048],
            trainable=False,
        )(self._inputs)

        # Classification head
        self._class_dense1 = tf.keras.layers.Dense(1024, activation="relu")(
                self._feature_extractor
            )
        self._class_dense2 = tf.keras.layers.Dense(128, activation="relu")(
                self._class_dense1
            )
        
        self._num_classes = n_classes  # 20 in the VOC dataset
        self._classification_head = tf.keras.layers.Dense(
            self._num_classes,
            use_bias=False,
            name="output_1")(
                self._class_dense2
            )

        # Regression head
        self._reg_dense1 = tf.keras.layers.Dense(512, activation="relu")(
            self._feature_extractor
            )
        self._reg_coordinates = tf.keras.layers.Dense(4, name="output_2", use_bias=False)(
                self._reg_dense1
            )

        self._model = tf.keras.Model(
                inputs=self._inputs,
                outputs=[self._classification_head, self._reg_coordinates],
                name="simple_od_net"
            )
    
    def __call__(self, inputs, training=False):  # training arg may be needed if dropout is used
        # x = self._feature_extractor(inputs)
        # x = self._reg_dense1(x)
        # coordinates = self._reg_coordinates(x)
        # x = self._class_dense1(x)
        # x = self._class_dense2(x)
        # classification_head = self._classification_head(x)

        return self._model(inputs)


# First option -> this requires to call the loss l2, taking care of squeezing the input
# l2 = tf.losses.MeanSquaredError()

# Second option, it is the loss function iself that squeezes the input
def l2(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - tf.squeeze(y_true, axis=[1])))


precision_metric = tf.metrics.Precision()


def iou(pred_box, gt_box, h, w):
    """
    Compute IoU between detect box and gt boxes
    Args:
        pred_box: shape (4, ):  y_min, x_min, y_max, x_max - predicted box
        gt_boxes: shape (n, 4): y_min, x_min, y_max, x_max - ground truth
        h: image height
        w: image width
    """

    # Transpose the coordinates from y_min, x_min, y_max, x_max
    # In absolute coordinates to x_min, y_min, x_max, y_max
    # in pixel coordinates
    def _swap(box):
        return tf.stack([box[1] * w, box[0] * h, box[3] * w, box[2] * h])

    pred_box = _swap(pred_box)
    gt_box = _swap(gt_box)

    box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    xx1 = tf.maximum(pred_box[0], gt_box[0])
    yy1 = tf.maximum(pred_box[1], gt_box[1])
    xx2 = tf.minimum(pred_box[2], gt_box[2])
    yy2 = tf.minimum(pred_box[3], gt_box[3])

    # compute the width and height of the bounding box
    w = tf.maximum(0, xx2 - xx1)
    h = tf.maximum(0, yy2 - yy1)

    inter = w * h
    return inter / (box_area + area - inter)


threshold = 0.75


def draw(dataset, regressor, step):
    with tf.device("/CPU:0"):
        row = next(iter(dataset.take(3).batch(3)))
        images = row["image"]
        obj = row["objects"]
        boxes = regressor(images)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        images = tf.image.draw_bounding_boxes(
            images=images, boxes=tf.reshape(boxes, (-1, 1, 4)), colors=colors
        )
        images = tf.image.draw_bounding_boxes(
            images=images, boxes=tf.reshape(obj["bbox"], (-1, 1, 4)), colors=colors
        )
        tf.summary.image("images", images, step=step)

        true_labels, predicted_labels = [], []
        for idx, predicted_box in enumerate(boxes):
            iou_value = iou(predicted_box, tf.squeeze(obj["bbox"][idx]), 299, 299)
            true_labels.append(1)
            predicted_labels.append(1 if iou_value >= threshold else 0)

        precision_metric.update_state(true_labels, predicted_labels)
        tf.summary.scalar("precision", precision_metric.result(), step=step)
        tf.print(precision_metric.result())


# Define a Object Detection Trainer with
# - a init method defining the model, loss, optimizer, metrics
#    we require that the multiple outputs of the model are named to be able to map appropriately loss, metrics to the right outputs
# - a train_step method that computes the loss and applies the gradients
#    the train method compile the model by defining the mapping between the outputs and the loss, metrics, and the ponderation of the loss
class ObjectDetectionTrainer:
    def __init__(self, model):
        self.model = model
        self.classification_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.regression_loss = l2
        self.optimizer = tf.optimizers.Adam(1e-3)
        # self.accuracy = tf.metrics.Accuracy()
        # self.mean_loss = tf.metrics.Mean(name='loss')
        self.model.compile(
                optimizer=self.optimizer,
                loss={
                    "output_1": self.classification_loss,
                    "output_2": self.regression_loss
                },
                loss_weights={
                    "output_1": 1.0,
                    "output_2": 1.0
                },
                metrics={
                    "output_1": "accuracy",
                    "output_2": "mse"
                }
            )
    
    # def train(self, train_x, train_labels, train_coord, 
    #         validation_x, validation_labels, validation_coord,
    #         epochs=10, batch_size=32):
    def train(self, ds_train, 
            ds_validation,
            epochs=10, batch_size=32):
        # metrics and losses are traced at the frequency update_freq
        # update_freq can be "batch" or "epoch", if "batch" is chosen, the metrics are traced at each batch
        # this can be customised by overwriting the method train_step
        logdir = "log/double_head/"
        tb_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
        checkpoints_filepath = "double_head_ckpts/checkpoint.model.keras"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_filepath, 
            # save_weights_only=True,
            monitor = "val_loss",
            mode = "auto",
            save_freq='epoch')
        # training_history = self.model.fit(train_x, [train_labels, train_coord], 
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     callbacks=[cp_callback, tb_callback],
        #     validation_data=(validation_x, [validation_labels, validation_coord]),
        #     verbose=1 # progress bar
        #     )
        training_history = self.model.fit(ds_train, 
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[cp_callback, tb_callback],
            validation_data=ds_validation,
            validation_steps=ds_validation.cardinality().numpy()//batch_size,
            verbose=1 # progress bar
            )
        

def train():
    n_classes = 20
    nepochs = 10
    batch_size = 32
    # Prepare datasets
    train, test, validation = prepare_datasets(batch_size=batch_size)

    # Define the model
    model = MObjectDetector(n_classes)

    trainer = ObjectDetectionTrainer(model)
    
    trainer.train(
        # train, validation,
        ds_train=train.map(lambda x: (x["image"], {"output_1": x["objects"]["label"], "output_2": x["objects"]["bbox"]})),
        ds_validation=validation.map(lambda x: (x["image"], {"output_1": x["objects"]["label"], "output_2": x["objects"]["bbox"]})),
        epochs=nepochs, batch_size=batch_size
    )

if __name__ == "__main__":
    train()
