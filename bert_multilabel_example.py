

import os

from critical_path.BERT.configs import ConfigClassifier
from critical_path.BERT.model_multilabel_class import (
    DataProcessor,
    MultiLabelClassifier)

import random
import pandas as pd
import tensorflow as tf


def read_toxic_data(randomize=False):

    df = pd.read_csv('../data/multi_class/train.csv')
    label_list = ["toxic", "severe_toxic", "obscene",
                  "threat", "insult", "identity_hate"]

    input_ids = []
    input_text = []
    input_labels = []

    for _, row in df.iterrows():
        sample_labels = []
        for label in label_list:
            if row[label] == 1:
                sample_labels.append(label)

        input_ids.append(row['id'])
        input_text.append(row['comment_text'])
        input_labels.append(sample_labels)

    if randomize:
        zipped = list(zip(input_ids, input_text, input_labels))
        random.shuffle(zipped)
        input_ids, input_text, input_labels = zip(*zipped)

    return input_ids, input_text, input_labels, label_list


def read_test_data():

    df = pd.read_csv('../data/multi_class/test.csv')
    label_list = ["toxic", "severe_toxic", "obscene",
                  "threat", "insult", "identity_hate"]

    input_ids = []
    input_text = []
    input_labels = []

    for _, row in df.head(25).iterrows():
        input_ids.append(row['id'])
        input_text.append(row['comment_text'])
        input_labels.append([])

    return input_ids, input_text, input_labels, label_list


def bert_multilabel(do_train=False, do_eval=False, do_predict=False):

    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/toxic_comment"

    data_dir = '../data/multi_class/'

    Flags = ConfigClassifier()
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        data_dir=data_dir)

    Flags.set_model_params(
        batch_size_train=8,
        max_seq_length=256,
        num_train_epochs=3)

    # Create new model
    FLAGS = Flags.get_handle()
    model = MultiLabelClassifier(FLAGS)

    if do_train:
        # Load data
        input_ids, input_text, input_labels, label_list = read_toxic_data(
            randomize=True)

        processor = DataProcessor(label_list=label_list)
        train_examples = processor.get_samples(
            input_ids=input_ids,
            input_text=input_text,
            input_labels=input_labels,
            set_type='train')

        model.train(train_examples, label_list)

    if do_eval:
        input_ids, input_text, input_labels, label_list = read_toxic_data()

        processor = DataProcessor(label_list=label_list)
        eval_examples = processor.get_samples(
            input_ids=input_ids,
            input_text=input_text,
            input_labels=input_labels,
            set_type='eval')

        results = model.eval(eval_examples, label_list)

        # Write results
        output_eval_file = os.path.join(FLAGS.bert_output_dir,
                                        "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(results.keys()):
                tf.logging.info("  %s = %s", key, str(results[key]))

    if do_predict:
        input_ids, input_text, input_labels, label_list = read_test_data()

        processor = DataProcessor(label_list=label_list)
        predict_examples = processor.get_samples(
            input_ids=input_ids,
            input_text=input_text,
            input_labels=input_labels,
            set_type='predict')

        results = model.predict(predict_examples, label_list)

        output = {k: [] for k in label_list}
        output['id'] = []

        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(results):
            if i % 10 == 0:
                print(i, f'--> {(i/len(results))*100}%')
            probabilities = prediction["probabilities"]
            output['id'].append(input_ids[i])
            for l in range(len(probabilities)):
                output[label_list[l]].append(probabilities[l])

        res = pd.DataFrame(output)
        res.to_csv('../data/multi_class/submission.csv',
                   index=False,
                   columns=["id", "toxic", "severe_toxic", "obscene",
                            "threat", "insult", "identity_hate"])


if __name__ == '__main__':

    bert_multilabel(do_train=False,
                    do_eval=False,
                    do_predict=True)
