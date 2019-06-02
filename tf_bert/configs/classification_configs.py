
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


# Task configuration
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")


# Classification specific files
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


# Define classification task
flags.DEFINE_string("task_name", None, "The name of the task to train.")


# Training Parameter
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
