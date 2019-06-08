

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os


from critical_path.BERT.configs import ConfigSQuAD
from critical_path.BERT.squad import read_squad_examples, write_squad_predictions

import critical_path.BERT.tokenization as tokenization

#from critical_path.BERT.modeling import BertConfig
from critical_path.BERT.squad import SQuADModel
from critical_path.BERT.squad import input_fn_builder, model_fn_builder
#from critical_path.BERT.squad import train_squad, eval_squad


import six
import tensorflow as tf

from datetime import datetime


def train_on_squad():

    Flags = ConfigSQuAD()

    base_model_folder_path = "C:/Users/Angus/models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/trained"
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path)

    Flags.set_model_params(
        batch_size_train=4,
        max_seq_length=384,
        max_answer_length=30,
        is_squad_v2=True,
        num_train_epochs=1)


    FLAGS = Flags.get_handle()
    model = SQuADModel(FLAGS)

    squad_train_path = "C:\\Users\\Angus\\data\\SQuAD_2.0\\small-train-2.0.json"
    train_samples = read_squad_examples(
        input_file=squad_train_path,
        is_training=True,
        is_squad_v2=FLAGS.is_squad_v2)

    model.train(train_samples) #, FLAGS)


def eval_on_squad():
    Flags = ConfigSQuAD()
    
    base_model_folder_path = "C:/Users/Angus/models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/qa_model"
    # restores model from highest checkpoint in directory
    
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path)

    Flags.set_model_params(is_squad_v2=True)


    FLAGS = Flags.get_handle()
    model = SQuADModel(FLAGS)

    pred_file = "C:\\Users\\Angus\\data\\SQuAD_2.0\\small-train-2.0.json"
    eval_samples = read_squad_examples(
        input_file=pred_file,
        is_training=False,
        is_squad_v2=FLAGS.is_squad_v2)

    results = model.predict(eval_samples)


    # Save results
    output_prediction_file = os.path.join(
        FLAGS.bert_output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        FLAGS.bert_output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(
        FLAGS.bert_output_dir, "null_odds.json")

    eval_features = results['eval_features']
    all_results = results['all_results']

    write_squad_predictions(
        eval_samples, eval_features, all_results,
        FLAGS.n_best_size, FLAGS.max_answer_length,
        FLAGS.do_lower_case, output_prediction_file,
        output_nbest_file, output_null_log_odds_file,
        is_squad_v2=FLAGS.is_squad_v2,
        null_score_diff_threshold=FLAGS.null_score_diff_threshold)


if __name__ == '__main__':
    #train_on_squad()
    eval_on_squad()


