# critical-path-nlp
Tools for adapting universal language models to specifc tasks

Please note these are not tools for hyper parameter tuning. 
Rather, these are tools for rapid prototyping to figure out the best model in the best problem frame for the task at hand.

Adapted from:
[https://github.com/google-research/bert]

# Current Capabilities

## BERT for SQuAD

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
  

