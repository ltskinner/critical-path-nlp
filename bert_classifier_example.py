

import os

from critical_path.BERT.configs import ConfigClassifier

import critical_path.BERT.tokenization as tokenization
import critical_path.BERT.modeling as modeling

from critical_path.BERT.classifier import model_fn_builder
from critical_path.BERT.classifier import file_based_convert_examples_to_features
from critical_path.BERT.classifier import file_based_input_fn_builder
from critical_path.BERT.classifier import ColaProcessor

import tensorflow as tf


from critical_path.BERT.classifier import OneLabelColumnProcessor
from critical_path.BERT.classifier import PaddingInputExample
from critical_path.BERT.classifier import ClassifierModel

import pandas as pd


def read():
    data_dir = "../data/class_data/"

    data = pd.read_csv(os.path.join(data_dir, "train.csv"))

    input_ids = data.s_id
    input_text = data.text
    input_labels = data.label

    # label_list = ["label_0", "label_1"]
    label_list = input_labels.value_counts()

    print(data.head())
    
    processor = OneLabelColumnProcessor(label_list=label_list)

    results = processor.get_train_examples(
        input_ids=input_ids,
        input_text=input_text,
        input_labels=input_labels)


def train_classifier():
    # Set flags
    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "class_model"
    
    data_dir = '../data/class_data/cola_public/raw/'

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
        num_train_epochs=1)

    # Create new model
    FLAGS = Flags.get_handle()
    model = ClassifierModel(FLAGS)

    processor = ColaProcessor()
    train_samples = processor.get_train_examples(FLAGS.data_dir)
    label_list = processor.get_labels()

    model.train(train_samples, label_list)


def eval_classifier():

    # Set flags
    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "class_model"
    
    data_dir = '../data/class_data/cola_public/raw/'

    Flags = ConfigClassifier()
    # TODO: delete this handle from all configs
    # Flags.set_task(do_train=True)
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        data_dir=data_dir)

    Flags.set_model_params(
        batch_size_eval=4,  # Note, need to move to .predict() ?
        max_seq_length=384,
        num_train_epochs=1)

    # Create new model
    FLAGS = Flags.get_handle()
    model = ClassifierModel(FLAGS)

    processor = ColaProcessor()
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    label_list = processor.get_labels()

    results = model.eval(eval_examples, label_list)

    output_eval_file = os.path.join(FLAGS.bert_output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(results.keys()):
            tf.logging.info("  %s = %s", key, str(results[key]))


def predict_classifier():

    # Set flags
    base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"
    name_of_config_json_file = "bert_config.json"
    name_of_vocab_file = "vocab.txt"
    output_folder_path = base_model_folder_path + "class_model"
    
    data_dir = '../data/class_data/cola_public/raw/'

    Flags = ConfigClassifier()
    # TODO: delete this handle from all configs
    # Flags.set_task(do_train=True)
    Flags.set_model_paths(
        bert_config_file=base_model_folder_path + name_of_config_json_file,
        bert_vocab_file=base_model_folder_path + name_of_vocab_file,
        bert_output_dir=output_folder_path,
        data_dir=data_dir)

    Flags.set_model_params(
        batch_size_predict=4,  # not sure the difference bw this and eval
        max_seq_length=384,
        num_train_epochs=1)

    # Create new model
    FLAGS = Flags.get_handle()
    model = ClassifierModel(FLAGS)

    processor = ColaProcessor()
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    label_list = processor.get_labels()

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


if __name__ == "__main__":
    #read()
    #train_classifier()
    #eval_classifier()
    predict_classifier()

    print("[!] uncouple .get_{}_examples()")
    print("... convert to get_examples(file_path, 'dev')")
    print("... use if 'dev' --> 'eval'")

    print("[+] Remember to pep8 stuff")
