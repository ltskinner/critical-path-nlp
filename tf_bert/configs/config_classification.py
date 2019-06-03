
import tensorflow as tf

# flags = tf.flags
# FLAGS = flags.FLAGS


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



def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict'" +
            " must be True.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))


