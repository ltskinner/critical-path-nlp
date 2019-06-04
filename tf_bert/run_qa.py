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
import random

# Not sure if 'modeling' needs to be imported as a whole
from models.modeling import BertConfig
#import models.modeling as modeling

import models.optimization as optimization
import data.tokenization as tokenization
from data.squad_data import read_squad_examples, write_squad_predictions
from data.squad_data import FeatureWriter, convert_examples_to_features
from models.squad_model import input_fn_builder, model_fn_builder
from configs.config_squad import ConfigSQuAD

import six
import tensorflow as tf



# TODO: Load configs
FLAGS = None




# TODO: Expand read_squad_examples to take is_squad_v2 = FLAGS.


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

# TODO: Expand write_predicitons to take is_squad_v2 = FLAGS.xx
# Need to manage other flag dependencies

# TODO: First test just try and pred!!! Do NOT start a training session



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Base BERT model folder path
    base_model_folder_path = "C:/Users/Angus/models/uncased_L-12_H-768_A-12"

    # Standard BERT config file name
    name_of_config_json_file = "bert_config.json"

    # Standard BERT vocab file name
    name_of_vocab_file = "vocab.txt"

    # Folder for output to be written to
    output_folder_path = base_model_folder_path + "/trained"

    Flags = ConfigSQuAD()
    Flags.set_tpu_gpu()
    Flags.set_run_configs()

    Flags.set_task(
        # do_train=True,
        do_predict=True)

    Flags.set_paths(
        bert_config_file=name_of_config_json_file,
        bert_vocab_file=name_of_vocab_file,
        bert_output_dir=output_folder_path,

        # file_to_train=squad_10_path, # Need to figure out path vs folder
        file_to_predict=squad_test_path)

    Flags.set_training_params(
        max_seq_length=69,
        max_answer_length=45)

    Flags.validate_flags_and_config()
    # TODO: unglobalize this (if possible)
    FLAGS = Flags.flags.FLAGS


    # Configuration
    # TODO: Modify configuration specification
    #bert_config = modeling.BertConfig.from_json_file(
    #    run_flags.bert_config_file)
    bert_config = BertConfig.from_json_file(run_flags.bert_config_file)
    bert_config.validate_input_size(FLAGS)

    tokenization.validate_word_cases(FLAGS.do_lower_case, 
                                     FLAGS.init_checkpoint)


    tf.gfile.MakeDirs(FLAGS.bert_output_dir)




    # Start processing

    # TODO: Wrap TPU handling
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.bert_output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # TODO: Wrap data loading routine
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_squad_examples(
            input_file=FLAGS.file_to_train, is_training=True)
        num_train_steps = int(
            len(train_examples) /
            FLAGS.batch_size_train * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    # Model initialization
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.batch_size_train,
        predict_batch_size=FLAGS.batch_size_predict)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.bert_vocab_file, do_lower_case=FLAGS.do_lower_case)



    # TODO: Wrat this in data loading routine
    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant
        # tensors in memory.
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.bert_output_dir, "train.tf_record"),
            is_training=True)
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size_train)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)







    # TODO: Wrap prediction routine
    if FLAGS.do_predict:
        eval_examples = read_squad_examples(
            input_file=FLAGS.file_to_predict, is_training=False)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.bert_output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size_predict)

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))


        # Save results
        output_prediction_file = os.path.join(FLAGS.bert_output_dir,
                                              "predictions.json")
        output_nbest_file = os.path.join(FLAGS.bert_output_dir,
                                         "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(FLAGS.bert_output_dir,
                                                 "null_odds.json")

        write_squad_predictions(eval_examples, eval_features, all_results,
                                FLAGS.n_best_size, FLAGS.max_answer_length,
                                FLAGS.do_lower_case, output_prediction_file,
                                output_nbest_file, output_null_log_odds_file,
                                is_squad_v2=)





if __name__ == "__main__":
    flags.mark_flag_as_required("bert_vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("bert_output_dir")
    tf.app.run()
