
import json
import tensorflow as tf

input_file = "../data/SQuAD_2.0/train-v2.0.json"


with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)

small_json = {
    'version': input_data['version'],
    'data': []
}

data = []
for entry in input_data["data"][:2]:
    small_json['data'].append(entry)


output_file = "../data/SQuAD_2.0/small-train-2.0.json"
with open(output_file, 'w') as fp:
    json.dump(small_json, fp)
