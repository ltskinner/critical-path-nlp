

import os

from critical_path.BERT.configs import ConfigClassifier

#from critical_path.BERT.model_multilabel_class import ClassifierModel
from critical_path.BERT.model_multilabel_class import DataProcessor
from critical_path.BERT.model_multilabel_class import MultiLabelClassifier
#from critical_path.BERT.model_classifier import OneLabelColumnProcessor

import pandas as pd
import tensorflow as tf


def read_data():
    df = pd.read_csv('../data/multi_class/train.csv')

    label_list = ["toxic", "severe_toxic", "obscene", 
                  "threat", "insult", "identity_hate"]

    input_ids = []
    input_text = []
    input_labels = []

    for _, row in df.head(4000).iterrows():
        sample_labels = []
        for label in label_list:
            if row[label] == 1:
                sample_labels.append(label)

        input_ids.append(row['id'])
        input_text.append(row['comment_text'])
        input_labels.append(sample_labels)

    return input_ids, input_text, input_labels, label_list


def train_multilabel():
    # Set flags
    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/multi_class"

    data_dir = '../data/multi_class/'

    Flags = ConfigClassifier()
    # TODO: delete this handle from all configs
    # Flags.set_task(do_train=True)
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        data_dir=data_dir)

    Flags.set_model_params(
        batch_size_train=4,  # Move to .train() ?
        max_seq_length=384,
        num_train_epochs=4)

    # Create new model
    FLAGS = Flags.get_handle()

    input_ids, input_text, input_labels, label_list = read_data()

    processor = DataProcessor(label_list=label_list)
    train_examples = processor.get_samples(
        input_ids=input_ids,
        input_text=input_text,
        input_labels=input_labels,
        set_type='train')

    model = MultiLabelClassifier(FLAGS)

    model.train(train_examples, label_list)


def eval_multilabel():
    # Set flags
    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/multi_class"

    data_dir = '../data/multi_class/'

    Flags = ConfigClassifier()
    # TODO: delete this handle from all configs
    # Flags.set_task(do_train=True)
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        data_dir=data_dir)

    Flags.set_model_params(
        batch_size_train=4,  # Move to .train() ?
        max_seq_length=384,
        num_train_epochs=4)

    # Create new model
    FLAGS = Flags.get_handle()

    input_ids, input_text, input_labels, label_list = read_data()

    processor = DataProcessor(label_list=label_list)
    eval_examples = processor.get_samples(
        input_ids=input_ids,
        input_text=input_text,
        input_labels=input_labels,
        set_type='eval')

    model = MultiLabelClassifier(FLAGS)

    results = model.eval(eval_examples, label_list)

    output_eval_file = os.path.join(FLAGS.bert_output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(results.keys()):
            tf.logging.info("  %s = %s", key, str(results[key]))


def predict_multilabel():
    # Set flags
    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "/multi_class"

    data_dir = '../data/multi_class/'

    Flags = ConfigClassifier()
    # TODO: delete this handle from all configs
    # Flags.set_task(do_train=True)
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        data_dir=data_dir)

    Flags.set_model_params(
        batch_size_train=4,  # Move to .train() ?
        max_seq_length=384,
        num_train_epochs=4)

    # Create new model
    FLAGS = Flags.get_handle()

    input_ids, input_text, input_labels, label_list = read_data()

    processor = DataProcessor(label_list=label_list)
    predict_examples = processor.get_samples(
        input_ids=input_ids,
        input_text=input_text,
        input_labels=input_labels,
        set_type='eval')

    model = MultiLabelClassifier(FLAGS)

    results = model.predict(predict_examples, label_list)
    # Actually write the results
    # Basically, each column is the confidence for a label
    output_predict_file = os.path.join(FLAGS.bert_output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(results):
            probabilities = prediction["probabilities"]
            output_line = "\t".join(
                str(class_probability)
                for class_probability in probabilities) + "\n"
            writer.write(output_line)



if __name__ == '__main__':
    #read()
    #train_multilabel()
    #eval_multilabel()
    #predict_multilabel()


    tf.sigmoid([0, 1, 0, 1, 0, 0, 0])
