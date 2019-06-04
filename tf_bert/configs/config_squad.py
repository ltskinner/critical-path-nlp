
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

from .config_base import ConfigBase

# TODO: abstract 'predict_file' to not require offline file
# TODO: think about adding 'task_name' do .set_task() to mimic classification


class ConfigSQuAD(ConfigBase):
    """Configuration flags specific to SQuAD implementations of BERT"""
    def __init__(self,):
        super().__init__()

    def set_task(self,
                 do_train=False,
                 do_predict=False):
        """Set whether training or predicting"""
        self.flags.DEFINE_bool(
            "do_train", do_train,
            "Whether to run training.")

        self.flags.DEFINE_bool(
            "do_predict", do_predict,
            "Whether to run eval on the dev set.")

    def set_paths(self,
                  file_to_train=None,
                  file_to_predict=None,
                  *args, **kwargs):
        """Set SQuAD specific files, in addition to BERT files"""
        self.flags.DEFINE_string(
            "train_file", file_to_train,
            "SQuAD json for training. E.g., train-v1.1.json")

        self.flags.DEFINE_string(
            "predict_file", file_to_predict,
            "SQuAD json for predictions. " +
            "E.g., dev-v1.1.json or test-v1.1.json")

        ConfigBase.set_paths(self, *args, **kwargs)

    def set_run_configs(self,
                        verbose_logging=False,
                        is_squad_v2=False,
                        null_score_diff_threshold=0.0):
        """Set non technical SQuAD specific configurations"""
        self.flags.DEFINE_bool(
            "verbose_logging", verbose_logging,
            "If true, all of the warnings related to data processing will "
            "be printed. A number of warnings are expected for a normal "
            "SQuAD evaluation.")

        self.flags.DEFINE_bool(
            "is_squad_v2", is_squad_v2,
            "If true, the SQuAD examples contain some that do not have an "
            "answer.")

        self.flags.DEFINE_float(
            "null_score_diff_threshold", null_score_diff_threshold,
            "If null_score - best_non_null is greater than the threshold "
            "predict null")

    def set_training_params(self,
                            doc_stride=128,
                            max_query_length=64,
                            n_best_size=20,
                            max_answer_length=30,
                            *args, **kwargs):
        """Set SQuAD training params"""
        self.flags.DEFINE_integer(
            "doc_stride", doc_stride,
            "When splitting up a long document into chunks, how much stride "
            "to take between chunks.")

        self.flags.DEFINE_integer(
            "max_query_length", max_query_length,
            "The maximum number of tokens for the question. Questions "
            "longer than this will be truncated to this length.")

        self.flags.DEFINE_integer(
            "n_best_size", n_best_size,
            "The total number of n-best predictions to generate in the "
            "nbest_predictions.json output file.")

        self.flags.DEFINE_integer(
            "max_answer_length", max_answer_length,
            "The maximum length of an answer that can be generated. This is "
            "needed because the start and end predictions are not "
            "conditioned on one another")

        ConfigBase.set_training_params(self, *args, **kwargs)

    def use_defaults(self,):
        """Use all default params"""
        self.set_run_configs()
        self.set_training_params()

        ConfigBase.use_defaults(self)

    def validate_flags_and_config(self,):
        """Validate the input FLAGS or throw an exception."""
        FLAGS = self.flags.FLAGS

        if not FLAGS.do_train and not FLAGS.do_predict:
            raise ValueError("At least one of `do_train` or `do_predict` "
                             "must be True.")

        if FLAGS.do_train:
            if not FLAGS.train_file:
                raise ValueError(
                    "If `do_train` is True, then `train_file` must be "
                    "specified.")
        if FLAGS.do_predict:
            if not FLAGS.predict_file:
                raise ValueError(
                    "If `do_predict` is True, "
                    "then `predict_file` must be specified.")

        if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
            raise ValueError(
                "The max_seq_length (%d) must be greater "
                "than max_query_length (%d) + 3" %
                (FLAGS.max_seq_length, FLAGS.max_query_length))
