


"""
1) Load arguments

2) Error checking

3) Load tokenizer

4) If train

5) Load training examples

6) Load model

7) Initialize parallelization
    if fp16

8) Group optimizer parameters

9) Load Bert
    Bert is a 'BertAdam"
    takes the grouped optimizer parameters

10) If training
        try to load cached features file

11) Convert examples to features (bert function)
    Takes the training examples
    Takes the tokenizer
    Dump the training features

12) Create a bunch of tensors to hold the unput data

13) Slam all training tensors into a big tensor dataset

14) Randomize the training samples

15) Slam samples into a DataLoader

16) Enumerate through tqdm
    (this is comprable to tfrecords)
    a) Slam into model
    b) do loss and whatever

17) Save trained model
18) Save trianed tokenizer

------------------
19) If predict

20) Read testing examples

21) Convert examples to features

22) Slam into tensors

23) Slam tensors into data loaders

24) Iterate through tqdm

25) Collect results

26) Write predictions




"""


