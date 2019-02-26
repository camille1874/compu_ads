from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf



isbuyer = tf.feature_column.numeric_column("isbuyer")
buy_freq = tf.feature_column.numeric_column("buy_freq")
visit_freq = tf.feature_column.numeric_column("visit_freq")
buy_interval = tf.feature_column.numeric_column("buy_interval")
sv_interval = tf.feature_column.numeric_column("sv_interval")
expected_time_buy = tf.feature_column.numeric_column("expected_time_buy")
expected_time_visit = tf.feature_column.numeric_column("expected_time_visit")
last_buy = tf.feature_column.numeric_column("last_buy")
last_visit = tf.feature_column.numeric_column("last_visit")
multiple_buy = tf.feature_column.numeric_column("multiple_buy")
multiple_visit = tf.feature_column.numeric_column("multiple_visit")
uniq_urls = tf.feature_column.numeric_column("uniq_urls")
num_checkins = tf.feature_column.numeric_column("num_checkins")


# Wide columns and deep columns.
base_columns = [isbuyer, buy_freq, visit_freq, buy_interval, sv_interval, expected_time_buy, expected_time_visit, last_buy, last_visit, multiple_buy, multiple_visit, uniq_urls, num_checkins]

crossed_columns = [
    tf.feature_column.crossed_column(
        ["isbuyer", "buy_freq"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ["buy_interval", "last_buy"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ["buy_freq", "visit_freq"], hash_bucket_size=1000)
]

deep_columns = [isbuyer, buy_freq, visit_freq, buy_interval, sv_interval, expected_time_buy, expected_time_visit, last_buy, last_visit, multiple_buy, multiple_visit, uniq_urls, num_checkins]



def build_model(model_dir, model_type):
    if model_type == "wide":
        m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns + crossed_columns)
    elif model_type == "deep":
        m = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 50])
    else:
        m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
    return m


def read_data(data_file, num_epochs, shuffle):
    df_data = pd.read_csv(
    tf.gfile.Open(data_file),
    header=0,
    skipinitialspace=True,
    engine="python")
    labels = df_data["y_buy"]
    return tf.estimator.inputs.pandas_input_fn(
        x=df_data,
        y=labels,
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=5)



def train_and_eval(model_dir, model_type, train_steps, train_file_name, test_file_name):
    """Train and evaluate the model."""
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  
    m = build_model(model_dir, model_type)
    # set num_epochs to None to get infinite stream of data.
    m.train(
        input_fn=read_data(train_file_name, num_epochs=None, shuffle=True),
        steps=train_steps)
    # set steps to None to run evaluation until all data consumed.
    results = m.evaluate(
        input_fn=read_data(test_file_name, num_epochs=1, shuffle=False),
        steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    # Manual cleanup
    shutil.rmtree(model_dir)


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="./model/model_20000",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=20000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="./data/train.csv",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="./data/valid.csv",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
