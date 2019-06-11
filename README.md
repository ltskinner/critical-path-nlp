# critical-path-nlp
Tools for adapting universal language models to specifc tasks

Please note these are tools for rapid prototyping - not brute force hyperparameter tuning.

Adapted from: [Google's BERT](https://github.com/google-research/bert)

# Installation
1. Clone repo (just for now - will be on pip soon)
2. Download a pretrained [BERT model](https://github.com/google-research/bert#pre-trained-models) - start with **BERT-Base Uncased** if you're not sure where to begin
3. Unzip the model and make note of the path

# Examples
* **Full implementation examples can be found here:**
  + [**SQuAD example**](../master/bert_squad_example.py)
  + [**Multi-Label Classification example**](../master/bert_multilabel_example.py)
  + [**Single-Label Classification example**](../master/bert_classifier_example.py)

## Current Capabilities

### BERT for Question Answering

* Train and evaluate the SQuAD dataset
  + [SQuAD 2.0 - Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
  
### BERT for Multi-Label Classification

* Train and evaluate custom datasets for multi-label classification tasks (multiple labels possible)
  + [Kaggle - Google Toxic Comment Classification Challange](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### BERT for Single-Label Classification

* Train and evaluate custom datasets for single-label classification tasks (one label possible)
  + [CoLA - Corpus of Linguistic Acceptability](https://nyu-mll.github.io/CoLA/)
  + [MPRC - Microsoft Research Paraphrase Corpus](http://nlpprogress.com/english/semantic_textual_similarity.html)
  + [MNLI - Multi-Genre Natural Language Inference](https://www.nyu.edu/projects/bowman/multinli/)
  + etc.
  
## Future Capabilities

### BERT Word and Document Embeddings

### GPT-2 Training and Generation
  
# Usage + Core Components
### Configuring BERT
#### First, define the model paths


```python  

base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"  # Folder containing downloaded Base Model
name_of_config_json_file = "bert_config.json"  # Located inside the Base Model folder
name_of_vocab_file = "vocab.txt"  # Located inside the Base Model folder

output_directory = "../models/trained_BERT/" # Trained model and results landing folder

# Multi-Label and Single-Label Specific
data_dir = None  # Directory .tsv data is stored in - typically for CoLA/MPRC or other datasets with known structure

```

#### Second, define the model run parameters

```python

"""Settable parameters and their default values

Note: Most default values are perfectly fine
"""

# Administrative
init_checkpoint = None
save_checkpoints_steps = 1000
iterations_per_loop = 1000
do_lower_case = True   

# Technical
batch_size_train = 32
batch_size_eval = 8
batch_size_predict = 8
num_train_epochs = 3.0
max_seq_length = 128
warmup_proportion = 0.1
learning_rate = 3e-5

# SQuAD Specific
doc_stride = 128
max_query_length = 64
n_best_size = 20
max_answer_length = 30
is_squad_v2 = False  # SQuAD 2.0 has examples with no answer, aka "impossible", SQuAD 1.0 does not
verbose_logging = False
null_score_diff_threshold = 0.0

```
#### Initialize the configuration handler
```python

from critical_path.BERT.configs import ConfigClassifier

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

# Retrieve a handle for the configs
FLAGS = Flags.get_handle()
```

**A single 1070GTX using BERT-Base Uncased can handle**

| Model             | max_seq_len | batch_size |
| ----------------- |:-----------:| ----------:|
| BERT-Base Uncased |     256     |      6     |
|        ...        |     384     |      4     |

For full batch size and sequence length guidelines see Google's [recommendations](https://github.com/google-research/bert#out-of-memory-issues)

### Using Configured Model
#### First, create a new model with the configured parameters
```python

"""For Multi-Label Classification"""

from critical_path.BERT.model_multilabel_class import MultiLabelClassifier

model = MultiLabelClassifier(FLAGS)

```

#### Second, load your data source
* SQuAD has dedicated dataloaders
  + **read_squad_examples(), write_squad_predictions()** in [/BERT/model_squad](../master/critical_path/BERT/model_squad.py)
* Multi-Label Classification has a generic dataloader
  + **DataProcessor** in [/BERT/model_multilabel_class](../master/critical_path/BERT/model_multilabel_class.py)
    + **Note:** This requires data labels to be in string format
    + ```python
      labels = [
          ["label_1", "label_2", "label_3"],
          ["label_2"]
      ]
      ```
* Single-Label Classification dataloaders
  + **ColaProcessor** is implemented in [/BERT/model_classifier](../master/critical_path/BERT/model_classifier.py)
  + More dataloader formats have been done by [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py)
  
```python

"""For Multi-Label Classification with a custom .csv reading function"""
from critical_path.BERT.model_multilabel_class import DataProcessor

# read_data is dataset specifc - see /bert_multilabel_example.py
input_ids, input_text, input_labels, label_list = read_toxic_data(randomize=True)

processor = DataProcessor(label_list=label_list)
train_examples = processor.get_samples(
        input_ids=input_ids,
        input_text=input_text,
        input_labels=input_labels,
        set_type='train')

```

#### Third, run your task
```python

"""Train and predict a Multi-Label Classifier"""

if do_train:
  model.train(train_examples, label_list)
  
if do_predict:
  model.predict(predict_examples, label_list)

```

# For full examples please see:
* **Full implementations:**
  + [**SQuAD example**](../master/bert_squad_example.py)
  + [**Multi-Label Classification example**](../master/bert_multilabel_example.py)
  + [**Single-Label Classification example**](../master/bert_classifier_example.py)

