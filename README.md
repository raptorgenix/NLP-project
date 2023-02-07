# NLP-project
This model classifies sentiment of a sentence based on training data (this is a sentiment analysis program). This was a program I completed for class in 2021. 
The file reads input from training data (with labels included for the type of sentiment corresponding to each sentence) and then uses the information about the language used in negative/positive sentiment sentences to analyze whether 
test data is either negative or positive. There is an aggregated classification probability from the summation of the word sentiment weights used to determine the sentiment of the sentence.
Attached is a writeup that explains more about the Naive Bayes model training process and how I modified the normal training with a pre-process that cleaned the data by eliminating punctuation and ignored upper/lower case.
If a word is present in one class and missing in another, Laplace smoothing is used to ensure the vocabulary is the same in both possible classifications.
The train_file.txt is used to train sentiment classification, the label_test_data.txt is the output file for the expected sentiment given test data. The improved_label_test_data utilizes the pre-processed version of the test data. HW3-testset.txt are the test sentences being analyzed for sentiment. The writeup discusses the measurements of accuracy on how this classification performed.
