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
from data_utils import read_data
from data_utils import read_data_with_sampling
import codecs

isbuyer = tf.feature_column.numeric_column("isbuyer")
buy_freq = tf.feature_column.categorical_column_with_hash_bucket("buy_freq", hash_bucket_size=1000, dtype=tf.int64)
visit_freq = tf.feature_column.categorical_column_with_hash_bucket("visit_freq", hash_bucket_size=1000, dtype=tf.int64)
buy_interval = tf.feature_column.numeric_column("buy_interval")
sv_interval = tf.feature_column.numeric_column("sv_interval")
expected_time_buy = tf.feature_column.numeric_column("expected_time_buy")
expected_time_visit = tf.feature_column.numeric_column("expected_time_visit")
last_buy = tf.feature_column.categorical_column_with_hash_bucket("last_buy", hash_bucket_size=1000, dtype=tf.int64)
last_visit = tf.feature_column.categorical_column_with_hash_bucket("last_visit", hash_bucket_size=1000, dtype=tf.int64)
multiple_buy = tf.feature_column.numeric_column("multiple_buy")
multiple_visit = tf.feature_column.numeric_column("multiple_visit")
uniq_urls = tf.feature_column.categorical_column_with_hash_bucket("uniq_urls", hash_bucket_size=1000, dtype=tf.int64)
num_checkins = tf.feature_column.categorical_column_with_hash_bucket("num_checkins", hash_bucket_size=1000, dtype=tf.int64)


# Wide columns and deep columns.
base_columns = [isbuyer, buy_freq, visit_freq, last_buy, last_visit, multiple_buy, multiple_visit, uniq_urls, num_checkins]

crossed_columns = [
    tf.feature_column.crossed_column(["isbuyer", "buy_freq"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["buy_freq", "visit_freq"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["buy_interval", "sv_interval"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["expected_time_buy", "expected_time_visit"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["last_buy", "last_visit"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["uniq_urls", "num_checkins"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["visit_freq", "last_visit", ], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["buy_freq", "last_buy", ], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["buy_freq", "expected_time_buy", "last_buy", "multiple_buy"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["visit_freq", "expected_time_visit", "last_visit", "multiple_visit"], hash_bucket_size=1000)
]

deep_columns = [isbuyer, 
tf.feature_column.embedding_column(buy_freq, dimension=4),
tf.feature_column.embedding_column(visit_freq, dimension=8),
 buy_interval, sv_interval, expected_time_buy, expected_time_visit, 
tf.feature_column.embedding_column(last_buy, dimension=8),
tf.feature_column.embedding_column(last_visit, dimension=8),
tf.feature_column.embedding_column(uniq_urls, dimension=8),
tf.feature_column.embedding_column(num_checkins, dimension=16)]


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
        linear_feature_columns=base_columns + crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
    return m




def train_and_eval(model_dir, model_type, train_steps, train_file_name, valid_file_name, test_file_name, result_file):
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  
    m = build_model(model_dir, model_type)
    # set num_epochs to None to get infinite stream of data.
    rf = codecs.open(result_file, mode='w', encoding='utf-8')
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth=True
    with tf.Session(config=session_config) as sess:
        #m.train(input_fn=read_data(train_file_name, num_epochs=None, shuffle=True), steps=train_steps)
        m.train(input_fn=read_data_with_sampling(train_file_name, num_epochs=None, shuffle=True), steps=train_steps)
        eval_result = m.evaluate(input_fn=read_data(valid_file_name, num_epochs=1, shuffle=False), steps=None)
        print("model directory = %s" % model_dir)
        for key in sorted(eval_result):
            print("%s: %s" % (key, eval_result[key]))
        predictions = m.predict(input_fn=read_data(test_file_name, num_epochs=1, shuffle=False), predict_keys="classes")
        predictions = list(predictions)
        for p in predictions:
            rf.write(str(p["classes"][0] ,encoding='utf-8'))
            rf.write("\n")
        


FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir + "_" + FLAGS.model_type, FLAGS.model_type, FLAGS.train_steps, FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data, FLAGS.model_type + "_" + FLAGS.result_file)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model/model",
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
        default=5000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="./data/train.csv",
        help="Path to the training data."
    )
    parser.add_argument(
        "--valid_data",
        type=str,
        default="./data/valid.csv",
        help="Path to the valid data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/ads_test.csv",
        #default="./data/valid.csv",
        help="Path to the test data."
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="result.csv",
        help="Path to the result data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
