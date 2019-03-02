import tensorflow as tf
import pandas as pd

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
session_conf.gpu_options.allow_growth=True
sess = tf.Session(config=session_conf)
with sess.as_default():
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    
