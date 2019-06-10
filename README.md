# critical-path-nlp
Tools for adapting universal language models to specifc tasks

Please note these are not tools for hyper parameter tuning. 
Rather, these are tools for rapid prototyping to figure out the best model in the best problem frame for the task at hand.

Adapted from: [Google's BERT](https://github.com/google-research/bert)


# Current Capabilities

## BERT for Question Answering

* Train and evaluate the SQuAD dataset
  + [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
  
## BERT for Multi-Label Classification

* Train and evaluate custom datasets for multi-label classification tasks (multiple labels possible)
* Examples:
  + [Kaggle - Google Toxic Comment Classification Challange](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## BERT for Single-Label Classification

* Train and evaluate custom datasets for single-label classification tasks (one label possible)
* Examples:
  + [CoLA - Corpus of Linguistic Acceptability](https://nyu-mll.github.io/CoLA/)
  + [MPRC - Microsoft Research Paraphrase Corpus](http://nlpprogress.com/english/semantic_textual_similarity.html)
  + [MNLI - Multi-Genre Natural Language Inference](https://www.nyu.edu/projects/bowman/multinli/)
  + etc...
  

# Usage

## Installation
1. Clone repo (just for now - will be on pip soon)
2. Download a pretrained [BERT model](https://github.com/google-research/bert#pre-trained-models) - start with **BERT-Base Uncased** if youre not sure where to begin
3. Unzip the model and make note of the path

## Examples
* Code:
  + [SQuAD example](..blob/master/bert_squad_example.py)
  + [Multi-Label Classification example](../blob/master/bert_multilabel_example.py)
  + [Single-Label Classification example](../blob/master/bert_classifier_example.py)
  
### Core Components
#### Configuring BERT
* There are two main configurations that need to be done:
  + 1) Paths to models and support files
    + BERT Base model - the model downloaded from Google
      + bert_config.json
      + vocab.txt
    + Output Directory
      + Trained models are saved here
      + Written predictions land here
  + 2) Model Parameters
    + sdsd




