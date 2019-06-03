
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


class ConfigBase(object):
    """Configuration flags required in all BERT implementations"""

    def __init__(self,):
        self.flags = tf.flags

    def set_paths(self,
                  bert_config_file=None,
                  bert_vocab_file=None,
                  bert_output_dir=None,
                  *args, **kwargs):
        """Paths to support files required for BERT to run"""
        self.flags.DEFINE_string(
            "bert_config_file", bert_config_file,
            "The config json file corresponding to the pre-trained BERT " +
            "model. This specifies the model architecture.")

        self.flags.DEFINE_string(
            "vocab_file", bert_vocab_file,
            "The vocabulary file that the BERT model was trained on.")

        self.flags.DEFINE_string(
            "output_dir", bert_output_dir,
            "The output directory where the model checkpoints will be written")

    def set_training_params(self,
                            init_checkpoint=None,
                            do_lower_case=True,
                            max_seq_length=128,
                            warmup_proportion=0.1,
                            save_checkpoints_steps=1000,
                            iterations_per_loop=1000,
                            num_train_epochs=3.0,
                            train_batch_size=32,
                            predict_batch_size=8,
                            learning_rate=5e-5):
        """BERT model parameters"""
        self.flags.DEFINE_string(
            "init_checkpoint", init_checkpoint,
            "Initial checkpoint (usually from a pre-trained BERT model).")

        self.flags.DEFINE_bool(
            "do_lower_case", do_lower_case,
            "Whether to lower case the input text. Should be True for uncased "
            "models and False for cased models.")

        self.flags.DEFINE_integer(
            "max_seq_length", max_seq_length,
            "The maximum total input sequence length after WordPiece "
            "tokenization. Sequences longer than this will be truncated, "
            "and sequences shorter than this will be padded.")

        self.flags.DEFINE_float(
            "warmup_proportion", warmup_proportion,
            "Proportion of training to perform linear learning rate warmup "
            "for. E.g., 0.1 = 10% of training.")

        self.flags.DEFINE_integer(
            "save_checkpoints_steps", save_checkpoints_steps,
            "How often to save the model checkpoint.")

        self.flags.DEFINE_integer(
            "iterations_per_loop", iterations_per_loop,
            "How many steps to make in each estimator call.")

        self.flags.DEFINE_float(
            "num_train_epochs", num_train_epochs,
            "Total number of training epochs to perform.")

        self.flags.DEFINE_integer(
            "train_batch_size", train_batch_size,
            "Total batch size for training.")

        self.flags.DEFINE_integer(
            "predict_batch_size", predict_batch_size,
            "Total batch size for predict.")

        self.flags.DEFINE_float(
            "learning_rate", learning_rate,
            "The initial learning rate for Adam.")

    def set_tpu_gpu(self,
                    use_tpu=False,
                    tpu_name=None,
                    tpu_zone=None,
                    gcp_project=None,
                    master=None,
                    num_tpu_cores=8):
        """TPU or GPU/CPU Configuration"""
        self.flags.DEFINE_bool(
            "use_tpu", use_tpu,
            "Whether to use TPU or GPU/CPU.")

        self.flags.DEFINE_string(
            "tpu_name", tpu_name,
            "The Cloud TPU to use for training. This should be either the "
            "name used when creating the Cloud TPU, or a "
            "grpc://ip.address.of.tpu:8470 url.")

        self.flags.DEFINE_string(
            "tpu_zone", tpu_zone,
            "[Optional] GCE zone where the Cloud TPU is located in. If not "
            "specified, we will attempt to automatically detect the GCE "
            "project from metadata.")

        self.flags.DEFINE_string(
            "gcp_project", gcp_project,
            "[Optional] Project name for the Cloud TPU-enabled project. If "
            "not specified, we will attempt to automatically detect the GCE "
            "project from metadata.")

        self.flags.DEFINE_string(
            "master", master,
            "[Optional] TensorFlow master URL.")

        self.flags.DEFINE_integer(
            "num_tpu_cores", num_tpu_cores,
            "Only used if `use_tpu` is True. Total number of TPU cores to use")
