
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

# TODO: abstract '{train}_{eval}_{pred}_file' to not require offline file
# TODO: ensure this is a complete interface... seems light


class ConfigClassification(ConfigBase):
    """Configuration flags specific to Classification implementations of BERT
    """
    def __init__(self,):
        super().__init__()
        print("[!] ConfigGlass: Need to expand .set_task(), and modify to be")
        print("... more like ConfigSQuAD handles")
        print("[!] Need to change .set_path() to accept file not directory")
        print("[!] NEed to modify for offline processing")

    def set_task(self,
                 do_eval=False,
                 do_predict=False):
        """Set whether training, evaluating, or predicting"""

        # Define classification task
        self.flags.DEFINE_string(
            "task_name", None,
            "The name of the task to train.")

        # Task configuration
        self.flags.DEFINE_bool(
            "do_eval", do_eval,
            "Whether to run eval on the dev set.")

        self.flags.DEFINE_bool(
            "do_predict", do_predict,
            "Whether to run the model in inference mode on the test set.")

    def set_paths(self,
                  data_dir=None):
        # Classification specific files
        self.flags.DEFINE_string(
            "data_dir", None,
            "The input data dir. Should contain the .tsv files "
            "(or other data files) for the task.")

    def set_training_params(self,
                            eval_batch_size=8):
        # Training Parameter
        self.flags.DEFINE_integer(
            "eval_batch_size", eval_batch_size,
            "Total batch size for eval.")

    def validate_flags_or_throw(self,):
        """Validate the input FLAGS or throw an exception."""
        FLAGS = self.flags.FLAGS

        if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
            raise ValueError(
                "At least one of `do_train`, `do_eval` or `do_predict'" +
                " must be True.")
