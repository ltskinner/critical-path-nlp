

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os

from critical_path.BERT.configs import ConfigSQuAD
from critical_path.BERT.model_squad import (
    SQuADModel,
    input_fn_builder, model_fn_builder,
    read_squad_examples, write_squad_predictions)

import tensorflow as tf


def bert_squad(do_train=False, do_predict=False):
    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "squadl_model"

    Flags = ConfigSQuAD()
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path)

    Flags.set_model_params(batch_size_train=4,
                           max_seq_length=384,
                           max_answer_length=30,
                           num_train_epochs=1,
                           is_squad_v2=True)

    # Create new model
    FLAGS = Flags.get_handle()
    model = SQuADModel(FLAGS)

    if do_train:
        squad_train_path = "../data/SQuAD_2.0/small-train-2.0.json"
        train_samples = read_squad_examples(
            input_file=squad_train_path,
            is_training=True,
            is_squad_v2=FLAGS.is_squad_v2)

        model.train(train_samples)

    if do_predict:
        pred_file = "../data/SQuAD_2.0/small-train-2.0.json"
        eval_samples = read_squad_examples(
            input_file=pred_file,
            is_training=False,
            is_squad_v2=FLAGS.is_squad_v2)

        results = model.predict(eval_samples)

        # Save results
        eval_features = results['eval_features']
        all_results = results['all_results']

        output_folder = "../data/SQuAD_2.0/preds/"
        formatted_results = write_squad_predictions(
            eval_samples, eval_features, all_results,
            FLAGS.n_best_size, FLAGS.max_answer_length,
            FLAGS.do_lower_case, output_folder,
            is_squad_v2=FLAGS.is_squad_v2,
            null_score_diff_threshold=FLAGS.null_score_diff_threshold)


if __name__ == '__main__':
    bert_squad(do_train=True,
               do_predict=False)
