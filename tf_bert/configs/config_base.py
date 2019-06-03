

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


class ConfigBase(object):
    def __init__(self,
                 # File paths
                 bert_config_file=None,
                 vocab_file=None,
                 output_dir=None,

                 # Training parameters
                 init_checkpoint=None,
                 do_lower_case=True,
                 max_seq_length=128,
                 warmup_proportion=0.1,
                 save_checkpoints_steps=1000,
                 iterations_per_loop=1000,
                 num_train_epochs=3.0,
                 train_batch_size=32,
                 predict_batch_size=8,
                 learning_rate=5e-5,

                 # TPU configuration
                 use_tpu=False,
                 tpu_name=None,
                 tpu_zone=None,
                 gcp_project=None,
                 master=None,
                 num_tpu_cores=8):

        # BERT file paths
        flags.DEFINE_string(
            "bert_config_file", bert_config_file,
            "The config json file corresponding to the pre-trained BERT model."
            " This specifies the model architecture.")

        flags.DEFINE_string(
            "vocab_file", vocab_file,
            "The vocabulary file that the BERT model was trained on.")

        flags.DEFINE_string(
            "output_dir", output_dir,
            "The output directory where the model checkpoints will be written")

        # Model configs
        flags.DEFINE_string(
            "init_checkpoint", init_checkpoint,
            "Initial checkpoint (usually from a pre-trained BERT model).")

        flags.DEFINE_bool(
            "do_lower_case", do_lower_case,
            "Whether to lower case the input text. Should be True for uncased "
            "models and False for cased models.")

        flags.DEFINE_integer(
            "max_seq_length", max_seq_length,
            "The maximum total input sequence length after WordPiece " +
            "tokenization. Sequences longer than this will be truncated, " +
            "and sequences shorter than this will be padded.")

        flags.DEFINE_float(
            "warmup_proportion", warmup_proportion,
            "Proportion of training to perform linear learning rate warmup " +
            "for. E.g., 0.1 = 10% of training.")

        flags.DEFINE_integer(
            "save_checkpoints_steps", save_checkpoints_steps,
            "How often to save the model checkpoint.")

        flags.DEFINE_integer(
            "iterations_per_loop", iterations_per_loop,
            "How many steps to make in each estimator call.")

        # Training Parameters
        flags.DEFINE_float(
            "num_train_epochs", num_train_epochs,
            "Total number of training epochs to perform.")

        flags.DEFINE_integer(
            "train_batch_size", train_batch_size,
            "Total batch size for training.")

        flags.DEFINE_integer(
            "predict_batch_size",
            predict_batch_size, "Total batch size for predict.")

        flags.DEFINE_float(
            "learning_rate", learning_rate,
            "The initial learning rate for Adam.")

        # TPU or GPU/CPU Configuration
        flags.DEFINE_bool(
            "use_tpu", use_tpu,
            "Whether to use TPU or GPU/CPU.")

        tf.flags.DEFINE_string(
            "tpu_name", tpu_name,
            "The Cloud TPU to use for training. This should be either the " +
            "name used when creating the Cloud TPU, or a " +
            "grpc://ip.address.of.tpu:8470 url.")

        tf.flags.DEFINE_string(
            "tpu_zone", tpu_zone,
            "[Optional] GCE zone where the Cloud TPU is located in. If not " +
            "specified, we will attempt to automatically detect the GCE " +
            "project from metadata.")

        tf.flags.DEFINE_string(
            "gcp_project", gcp_project,
            "[Optional] Project name for the Cloud TPU-enabled project. If " +
            "not specified, we will attempt to automatically detect the GCE " +
            "project from metadata.")

        tf.flags.DEFINE_string(
            "master", master,
            "[Optional] TensorFlow master URL.")

        flags.DEFINE_integer(
            "num_tpu_cores", num_tpu_cores,
            "Only used if `use_tpu` is True. Total number of TPU cores to use")
