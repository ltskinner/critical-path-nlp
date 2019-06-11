# critical-path-nlp
Tools for adapting universal language models to specifc tasks

Please note these are not tools for hyper parameter tuning. 
Rather, these are tools for rapid prototyping to figure out the best model in the best problem frame for the task at hand.

Adapted from: [Google's BERT](https://github.com/google-research/bert)

# Installation
1. Clone repo (just for now - will be on pip soon)
2. Download a pretrained [BERT model](https://github.com/google-research/bert#pre-trained-models) - start with **BERT-Base Uncased** if youre not sure where to begin
3. Unzip the model and make note of the path

# Examples
* **Code:**
  + [SQuAD example](../master/bert_squad_example.py)
  + [Multi-Label Classification example](../master/bert_multilabel_example.py)
  + [Single-Label Classification example](../master/bert_classifier_example.py)

## Current Capabilities

### BERT for Question Answering

* Train and evaluate the SQuAD dataset
  + [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
  
### BERT for Multi-Label Classification

* Train and evaluate custom datasets for multi-label classification tasks (multiple labels possible)
  + [Kaggle - Google Toxic Comment Classification Challange](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### BERT for Single-Label Classification

* Train and evaluate custom datasets for single-label classification tasks (one label possible)
  + [CoLA - Corpus of Linguistic Acceptability](https://nyu-mll.github.io/CoLA/)
  + [MPRC - Microsoft Research Paraphrase Corpus](http://nlpprogress.com/english/semantic_textual_similarity.html)
  + [MNLI - Multi-Genre Natural Language Inference](https://www.nyu.edu/projects/bowman/multinli/)
  + etc.
  
  
## Core Components
### Configuring BERT
#### First, define the model paths


```python  

base_model_folder_path = "../models/uncased_L-12_H-768_A-12/"  # Folder containing downloaded Base Model
name_of_config_json_file = "bert_config.json"  # Inside the Base Model folder
name_of_vocab_file = "vocab.txt"  # Inside the Base Model folder

output_directory = "../models/trained_BERT/" # Trained model and results landing folder
```

#### Second, define the model run parameters

### Using Configured Model
#### First, create a new model with the configured parameters**

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
  



