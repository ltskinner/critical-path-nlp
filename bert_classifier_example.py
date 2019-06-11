

import os

from critical_path.BERT.configs import ConfigClassifier

from critical_path.BERT.model_classifier import ClassifierModel
from critical_path.BERT.model_classifier import ColaProcessor
from critical_path.BERT.model_classifier import OneLabelColumnProcessor


import tensorflow as tf
import pandas as pd


def custom_reader():
    data_dir = "../data/class_data/"

    data = pd.read_csv(os.path.join(data_dir, "train.csv"))

    input_ids = data.s_id
    input_text = data.text
    input_labels = data.label

    # label_list = ["label_0", "label_1"]
    label_list = input_labels.value_counts()

    print(data.head())

    processor = OneLabelColumnProcessor(label_list=label_list)

    training_examples = processor.get_train_examples(
        input_ids=input_ids,
        input_text=input_text,
        input_labels=input_labels)


def bert_classifier(do_train=False, do_eval=False, do_predict=False):
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

    if do_train:
        processor = ColaProcessor()
        train_samples = processor.get_train_examples(FLAGS.data_dir)
        label_list = processor.get_labels()

        model.train(train_samples, label_list)

    if do_eval:
        processor = ColaProcessor()
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        label_list = processor.get_labels()

        results = model.eval(eval_examples, label_list)

        output_eval_file = os.path.join(FLAGS.bert_output_dir,
                                        "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(results.keys()):
                tf.logging.info("  %s = %s", key, str(results[key]))

    if do_predict:
        processor = ColaProcessor()
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        label_list = processor.get_labels()

        results = model.predict(predict_examples, label_list)

        # Actually write the results
        # Basically, each column is the confidence for a label
        output_predict_file = os.path.join(FLAGS.bert_output_dir,
                                           "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(results):
                probabilities = prediction["probabilities"]
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)


if __name__ == "__main__":

    bert_classifier(do_train=False,
                    do_eval=False,
                    do_predict=False)
