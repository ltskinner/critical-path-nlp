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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os

# Not sure if 'modeling' needs to be imported as a whole
from models.modeling import BertConfig
# import models.modeling as modeling

import tokenization as tokenization
#from data.tokenization import validate_word_cases
from squad_data import read_squad_examples, write_squad_predictions
# from data.squad_data import FeatureWriter, convert_examples_to_features
from configs import ConfigSQuAD
from models.modeling import init_model

from models.squad_model import input_fn_builder, model_fn_builder
from models.squad_model import train_squad, eval_squad


import six
import tensorflow as tf

from datetime import datetime



RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits"])



# TODO: figure out file paths for reading trained model to make preds 


def trainer(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    base_model_folder_path = "C:/Users/Angus/models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/trained"
    squad_train_path = "C:\\Users\\Angus\\data\\SQuAD_2.0\\train-v2.0.json"

    # Configure the model and training session
    Flags = ConfigSQuAD()
    Flags.set_tpu_gpu()
    Flags.set_run_configs(is_squad_v2=True)

    Flags.set_task(do_train=True)

    Flags.set_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        file_to_train=squad_train_path)

    Flags.set_training_params(
        batch_size_train=4,
        max_seq_length=384,
        max_answer_length=30)

    Flags.validate_flags_and_config()
    FLAGS = Flags.flags.FLAGS

    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    bert_config.validate_input_size(FLAGS)
    tokenization.validate_word_cases(
        FLAGS.do_lower_case, FLAGS.init_checkpoint)

    tf.gfile.MakeDirs(FLAGS.bert_output_dir)

    # -------------------------------------------------------------------------
    
    train_samples = read_squad_examples(
        input_file=FLAGS.file_to_train,
        is_training=True,
        is_squad_v2=FLAGS.is_squad_v2)
    
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.bert_vocab_file, do_lower_case=FLAGS.do_lower_case)

    estimator = init_model(bert_config, FLAGS,
                           model_fn_builder=model_fn_builder,
                           train_samples=train_samples)

    if FLAGS.do_train:
        print('Training')
        train_squad(train_samples, estimator, tokenizer, FLAGS)
    
    """
    if FLAGS.do_predict:
        eval_samples = read_squad_examples(
            input_file=FLAGS.file_to_predict,
            is_training=False,
            is_squad_v2=FLAGS.is_squad_v2)

        results = eval_squad(eval_samples, estimator, tokenizer, FLAGS)

        eval_features = results['eval_features']
        all_results = results['eval_results']

        # Save results
        output_prediction_file = os.path.join(FLAGS.bert_output_dir,
                                              "predictions.json")
        output_nbest_file = os.path.join(FLAGS.bert_output_dir,
                                         "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(FLAGS.bert_output_dir,
                                                 "null_odds.json")

        write_squad_predictions(
            eval_samples, eval_features, all_results,
            FLAGS.n_best_size, FLAGS.max_answer_length,
            FLAGS.do_lower_case, output_prediction_file,
            output_nbest_file, output_null_log_odds_file,
            is_squad_v2=FLAGS.is_squad_v2,
            null_score_diff_threshold=FLAGS.null_score_diff_threshold)
    """


if __name__ == "__main__":
    #flags.mark_flag_as_required("bert_vocab_file")
    #flags.mark_flag_as_required("bert_config_file")
    #flags.mark_flag_as_required("bert_output_dir")
    #tf.app.run()
    startTime = datetime.now()
    main('')
    print("\nRun time:", datetime.now() - startTime)
