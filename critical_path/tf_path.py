
"""
1) Create bert_configuration object
        Loads in from json file

2) Create new tokenizer
        Needs vocab file

3) Determine if using TPUs

4) Decide if training
        a) Load squad examples
        b) Determine training steps
        c) Shuffle input data

5) Set the model function builder
        Takes bert config
        A checkpoint
        Leaning rate
        Some more crap

6) set the estimator

7) Create a file writer so large tensors dont have to be stored in memory

8) Convert training examples into features
        Takes train examples
        Takes the tokenizer
        Takes some other stuff
        Takes the file writing function

9) Create a data loading function to read the written tensors

10) Train the estimator on the data loading function

11) If predicting
    a) Read examples

12) Create feature writer
    a) Create list to hold features
    b) append each feature to the list
    c) write to disk

13) Convert examples to features
    Takes eval examples
    Takes tokenizer
    Takes feature writer

14) Create data loading function to read the written eval tensors

15) Iterate through estimator.predict()
    takes the eval data loading function

16) Custom SQuAD results parsing

17) Append named tuple raw result

18) Write results

"""
