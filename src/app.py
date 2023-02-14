import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

tfds.disable_progress_bar()

import argparse
import logging
import sys
import os
import json

BUFFER_SIZE = 100000


#Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.DEBUG)


# def parse_arguments(argv):

#   parser = argparse.ArgumentParser()
#   parser.add_argument('--log_dir', 
#                       type=str, 
#                       default='/logs',
#                       help='Name of the model folder.')
#   parser.add_argument('--train_steps',
#                       type=int,
#                       default=2,
#                       help='The number of training steps to perform.')
#   parser.add_argument('--batch_size',
#                       type=int,
#                       default=10,
#                       help='The number of batch size during training')
#   parser.add_argument('--learning_rate',
#                       type=float,
#                       default=0.01,
#                       help='Learning rate for training.')
#   parser.add_argument('--export_folder',
#                       type=float,
#                       help='folder to save model')

#   args, _ = parser.parse_known_args(args=argv[1:])

#   return args

# Katib parses metrics in this format: <metric-name>=<metric-value>.
class StdOutCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            "Epoch {:4d}: accuracy={:.4f} - loss={:.4f} \
                val_accuracy={:.4f} - val_loss={:.4f}".format(
                epoch+1, logs["accuracy"], logs["loss"],
                logs["val_accuracy"], logs["val_loss"]
            )
        )

def _is_chief(task_type, task_id):
    """Determines if the replica is the Chief."""
    return task_type is None or task_type == 'chief' or (
        task_type == 'worker' and task_id == 0)

def _scale(image, label):
    """Scales an image tensor."""
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label
  
def _get_saved_model_dir(base_path, task_type, task_id):
    """Returns a location for the SavedModel."""

    saved_model_path = base_path
    if not _is_chief(task_type, task_id):
        temp_dir = os.path.join('/tmp', task_type, str(task_id))
        tf.io.gfile.makedirs(temp_dir)
        saved_model_path = temp_dir

    return saved_model_path

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


def train(epochs, steps_per_epoch, per_worker_batch, checkpoint_path, saved_model_path):
    """Trains a MNIST classification model using multi-worker mirrored strategy."""

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    task_type = strategy.cluster_resolver.task_type
    task_id = strategy.cluster_resolver.task_id
    global_batch_size = per_worker_batch * strategy.num_replicas_in_sync

    with strategy.scope():
        datasets, _ = tfds.load(name='mnist', with_info=True, as_supervised=True)
        dataset = datasets['train'].map(_scale).cache().shuffle(BUFFER_SIZE).batch(global_batch_size).repeat()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        multi_worker_model = build_and_compile_cnn_model()

    callbacks = [
        tf.keras.callbacks.experimental.BackupAndRestore(checkpoint_path)
    ]

    multi_worker_model.fit(dataset, 
                           epochs=epochs, 
                           steps_per_epoch=steps_per_epoch,
                           callbacks=callbacks)

    
    logging.info("Saving the trained model to: {}".format(saved_model_path))
    saved_model_dir = _get_saved_model_dir(saved_model_path, task_type, task_id)
    multi_worker_model.save(saved_model_dir)


# def make_dataset():
#   # Load the data and split it between train and test sets
#   (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#   # Scale images to the [0, 1] range
#   x_train = x_train.astype("float32") / 255.
#   x_test = x_test.astype("float32") / 255.
#   # Make sure images have shape (28, 28, 1)
#   x_train = np.expand_dims(x_train, -1)
#   x_test = np.expand_dims(x_test, -1)

#   return x_train, y_train, x_test, y_test


def main(argv=None):
  logging.getLogger().setLevel(logging.INFO)
  tfds.disable_progress_bar()

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs',
                      type=int,
                      required=True,
                      help='Number of epochs to train.')
  parser.add_argument('--steps_per_epoch',
                      type=int,
                      required=True,
                      help='Steps per epoch.')
  parser.add_argument('--per_worker_batch',
                      type=int,
                      required=True,
                      help='Per worker batch.')
  parser.add_argument('--saved_model_path',
                      type=str,
                      required=True,
                      help='Tensorflow export directory.')
  parser.add_argument('--checkpoint_path',
                      type=str,
                      required=True,
                      help='Tensorflow checkpoint directory.')

  args = parser.parse_args()

  train(args.epochs, args.steps_per_epoch, args.per_worker_batch, 
      args.checkpoint_path, args.saved_model_path)


def eval_model(model, test_X, test_y):
  # evaluate the model performance
  score = model.evaluate(test_X, test_y, verbose=0)
  # if we can use the tf.event to collect metrics then we can display the 
  # train and validation curves for each trial
  print("accuracy={:2f}".format(score[1]))


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main()
