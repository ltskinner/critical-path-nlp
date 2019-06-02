
import os
import tensorflow as tf

BERT_FOLDER_PATH = "C:\\Users\\Angus\\models\\uncased_L-12_H-768_A-12"
BERT_CONFIG_JSON = "bert_config.json"
BERT_CKPT_NAME = "bert_model.ckpt"


def list_tf_architecture(model_ckpt_path):
    abs_ckpt_path = os.path.abspath(model_ckpt_path)
    model_variables_list = tf.train.list_variables(abs_ckpt_path)

    for variable, shape in model_variables_list:
        print(variable, shape)


if __name__ == "__main__":
    model_ckpt_path = BERT_FOLDER_PATH + "/" + BERT_CKPT_NAME
    list_tf_architecture(model_ckpt_path)
