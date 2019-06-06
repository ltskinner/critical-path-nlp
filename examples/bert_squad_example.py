

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os


from critical_path.bert.configs import ConfigSQuAD
from critical_path.bert.squad_data import read_squad_examples, write_squad_predictions

import critical_path.bert.tokenization as tokenization

from critical_path.bert.models.modeling import BertConfig, init_model
from critical_path.bert.models.squad_model import input_fn_builder, model_fn_builder
from critical_path.bert.models.squad_model import train_squad, eval_squad


import six
import tensorflow as tf

from datetime import datetime


def train_on_squad():
    tf.logging.set_verbosity(tf.logging.INFO)

    # Set required paths
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

    # Validate settings
    Flags.validate_flags_and_config()
    FLAGS = Flags.flags.FLAGS

    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    bert_config.validate_input_size(FLAGS)
    tokenization.validate_word_cases(
        FLAGS.do_lower_case, FLAGS.init_checkpoint)

    tf.gfile.MakeDirs(FLAGS.bert_output_dir)

    # Load training examples, initialize model and train
    train_samples = read_squad_examples(
        input_file=FLAGS.file_to_train,
        is_training=True,
        is_squad_v2=FLAGS.is_squad_v2)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.bert_vocab_file, do_lower_case=FLAGS.do_lower_case)

    estimator = init_model(bert_config, FLAGS,
                           model_fn_builder=model_fn_builder,
                           train_samples=train_samples)

    train_squad(train_samples, estimator, tokenizer, FLAGS)


def eval_on_squad():
    tf.logging.set_verbosity(tf.logging.INFO)

    # Set required paths
    base_model_folder_path = "C:/Users/Angus/models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/trained"
    squad_eval_path = "C:\\Users\\Angus\\data\\SQuAD_2.0\\train-v2.0.json"

    # Configure the model and training session
    Flags = ConfigSQuAD()
    Flags.set_tpu_gpu()
    Flags.set_run_configs(is_squad_v2=True)

    Flags.set_task(do_predict=True)

    Flags.set_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        file_to_predict=squad_eval_path)

    Flags.set_training_params(
        max_seq_length=384,
        max_answer_length=30)

    # Validate setting
    Flags.validate_flags_and_config()
    FLAGS = Flags.flags.FLAGS

    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    bert_config.validate_input_size(FLAGS)
    tokenization.validate_word_cases(
        FLAGS.do_lower_case, FLAGS.init_checkpoint)

    tf.gfile.MakeDirs(FLAGS.bert_output_dir)

    # Initialize model and predict
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.bert_vocab_file, do_lower_case=FLAGS.do_lower_case)

    estimator = init_model(bert_config, FLAGS,
                           model_fn_builder=model_fn_builder)

    eval_samples = read_squad_examples(
        input_file=FLAGS.file_to_predict,
        is_training=False,
        is_squad_v2=FLAGS.is_squad_v2)

    results = eval_squad(eval_samples, estimator, tokenizer, FLAGS)

    # Save results
    output_prediction_file = os.path.join(
        FLAGS.bert_output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        FLAGS.bert_output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(
        FLAGS.bert_output_dir, "null_odds.json")

    eval_features = results['eval_features']
    all_results = results['eval_results']

    write_squad_predictions(
        eval_samples, eval_features, all_results,
        FLAGS.n_best_size, FLAGS.max_answer_length,
        FLAGS.do_lower_case, output_prediction_file,
        output_nbest_file, output_null_log_odds_file,
        is_squad_v2=FLAGS.is_squad_v2,
        null_score_diff_threshold=FLAGS.null_score_diff_threshold)
