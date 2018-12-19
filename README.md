# Textfixer

The problem that this program is trying to solve is to fix the errors in the input text, and output the most probable correct sequence.
This is done to mimic the children's way of writing and making various types of mistakes and typos.

The idea is to use Baidu DeepSpeech algorithm with CTC loss to train the network to do symbol-based error correction.

Following training methodology has been chosen:

1) Train the network to copy the input into the output
2) Continue training the network with increasing difficulty: from fixing one symbol in one word to fixing a number of symbols in a number of words, to doing major changes in sentence structure (such as swapping the words or completing trimmed words).

The repository consists of two main workbooks:

1) Text distorter
2) Text fixer

## 1. Text distorter

This is a function that takes text as input and outputs distorted version of it. It is needed to add a parameter to specify, how many words in a sentencee need to be distorted.

## 2. Text fixer

The book uses an implementation of Baidu DeepSpeech algorithm, taken from: https://github.com/robmsmt/KerasDeepSpeech

Changes are following: 

1) the input dimensionality is equal to the size of the alphabet used (alphanumeric symbols, punctuation, some other symbols)
2) the output dimensionality is equal to the size of the alphabet as well
3) the type of input data is changed to one-hot vectors of text
4) the data used for training is Simple Wikipedia, divided into list of sentences, where sentences longer than 100 symbols or shorter than 20 symbols are removed

# Important notes
The package KENLM was found to be tricky to install on Windows, so I removed that part from the output. The N-gram model that was originally used in the DeepSpeech paper, needs to be trained separately.

The neural net actually runs the training, but CTC loss is quirky and I suspect that some part of output dimensionality might be causing the issue. See Tensorflow console output.
