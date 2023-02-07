# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
from nltk.stem import WordNetLemmatizer 
  
import re



"""
Your name and file comment here: Glenn Holzhauer

THIS FILE IS CURRENTLY BUILT TO YIELD THE INTENDED OUTPUT OF PRECISION, RECALL, AND F1 SCORE FROM THE DEV TEXT. 

"""


"""
Cite your sources here:

Presentation10-11_2.pptx lecture slides
https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
https://stackoverflow.com/questions/15092437/python-encoding-utf-8


"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""
#####################################################################################


def generate_tuples_from_file(training_file_path):
    datatuples = [] #the tuples, rows = lines and cols = 3 for ID, message, and positive/negative

    readfile = open(training_file_path, 'r', encoding='utf-8') ##character error came up with testset, google says it's encoding specific
    
    for line in readfile.readlines():
      
        tupleset = line.split('\t') ## tab as delimiter
        
        if (len(tupleset) == 3): ##check if test or train data, 3 = train and 2 = test

            datatuples.append((tupleset[0], tupleset[1], tupleset[2][:-1]))##to eliminate newline char 
        else:
            datatuples.append((tupleset[0], tupleset[1][:-1]))

    readfile.close()

    return datatuples





##################################################################################### so i can see where the functions are


def precision(gold_labels, classified_labels):
  
    truepos = 0
    falsepos = 0
  
    labelset = zip(gold_labels, classified_labels) #tuple for the label comparison of same line gold/classified
    for (reallabel, predicted) in labelset:
        if reallabel == "1": 
            if predicted == "1": 
                truepos = truepos + 1
          
        else: 
            if predicted == "1":
                falsepos = falsepos + 1
          
    return truepos / (truepos + falsepos)


#####################################################################################

def recall(gold_labels, classified_labels):
  
    truepos = 0
    falseneg = 0

    labelset = zip(gold_labels, classified_labels)
    for (reallabel, predicted) in labelset:
        if reallabel == "1":
            if predicted == "1": #correct prediction
                truepos = truepos + 1
            else: #expected 1 but got 0 
                falseneg = falseneg + 1

    return truepos / (truepos + falseneg)

#####################################################################################



def f1(gold_labels, classified_labels):

    rec = recall(gold_labels, classified_labels) 
    
    prec = precision(gold_labels, classified_labels)
  
    return (2*prec*rec) / (prec + rec) ##formula for f1



#####################################################################################
"""
Implement any other non-required functions here
"""

def predclass(sentanalysis, input): ##predict label for input
    predictions = []
    
    for dat in input:
        msg = dat[1]
        #print(msg)
        predictions.append(sentanalysis.classify(msg))
    return predictions

def reallabels(examples): #gets real labels from the devdata
    golds = []
    for (idtag, msg, senti) in examples:
        golds.append(senti)

    return(golds)

def writetofile(filepath, idtag, classification):
    writef = open(filepath, "a")
    towrite = idtag + " " + classification + '\n'
    writef.write(towrite)
    writef.close()

"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:


    def __init__(self):
    # do whatever you need to do to set up your class here
        self.poscount = 0      
        self.negcount = 0     
        self.posdict = {}
        self.negdict = {}
        self.positivewords = []  
        self.negativewords = []
   
        self.voc = []    

        self.probgivenclass = {}      #P(W|C) formula
        

    def train(self, examples):

        nummsg = len(examples)

        for i in range(nummsg):
            
            msg = examples[i][1]
            senti = examples[i][2]
            ##POPULATE TUPLES FOR NEGATIVE WORDS AND POSITIVE WORDS      
            if senti == '1': #keep track of overall pos/neg ratio

                self.positivewords.append(msg)
                self.poscount = self.poscount + 1
            else:
                self.negativewords.append(msg)
                self.negcount = self.negcount + 1

     

        for linepos, lineneg in zip(self.positivewords, self.negativewords):
            wordspos = linepos.split()
            wordsneg = lineneg.split()
            for word in wordspos:
                if word in self.posdict:
                    self.posdict[word] = self.posdict[word] + 1 #add 1 to how many times that word occurs in positive contexts
                else:
                    self.posdict[word] = 1 #or start at 1 because not yet seen
            for word in wordspos:
                if not(word in self.voc):
                    self.voc.append(word)
            for word in wordsneg:
                if word in self.negdict:
                    self.negdict[word] +=1
                else:
                    self.negdict[word] = 1
            for word in wordsneg:
                if not(word in self.voc):
                    self.voc.append(word)
                

                    
 
        self.chancepos = sum(self.posdict.values())
        self.chanceneg = sum(self.negdict.values())
        self.vsize = len(self.voc)
    
        nummsg = len(examples)

        #overall chance of neg or pos msg
        
        self.classchance = {}       #probability of class overall needed in formulas
        self.classchance['1'] = self.poscount/nummsg
        self.classchance['0'] = self.negcount/nummsg
        
        #find probability of word given class - LAPLACE SMOOTHING FORMULA FROM LECTURE
        for word in self.voc:
            if word in self.negdict:

                probbad = (self.negdict[word] + 1) / (self.chanceneg + self.vsize) #laplace again throughout
                self.probgivenclass[(word,'0')] = probbad ##updating probability 


            else:

                probbad = 1 / (self.chanceneg + self.vsize) #if word not found, handled with smoothing
                self.probgivenclass[(word,'0')] = probbad


        
            if word in self.posdict:

                probgood = (self.posdict[word] + 1) / (self.chancepos + self.vsize) #laplace smoothing formula! access dict for amount of occurrences of word
                self.probgivenclass[(word,'1')] = probgood
        

            else: 
                probgood = 1 / (self.chancepos + self.vsize) #this is done if the word is not found to avoid zeroing out
                self.probgivenclass[(word,'1')] = probgood
        
    
    def score(self, data): ##get chance it is positive or negative - calculate P(c)*P(data| c) 
        probgood = 0
        probbad = 0

        words = data.split()

        totalprobpos = 0.0 #sentence classing
        totalprobneg = 0.0
        
        for word in words:
            if word in self.voc: #check if its in the vocab; if it's not, just ignore it
                probgood += np.log(self.probgivenclass[(word,'1')])
                probbad += np.log(self.probgivenclass[(word,'0')])
            
                
        totalprobpos = np.exp(np.log(self.classchance['1']) + probgood)
        totalprobneg = np.exp(np.log(self.classchance['0']) + probbad)
        
        return { '1' : totalprobpos, '0' : totalprobneg}
                
                
    def classify(self, data):

        #create dict from score and see if 1 or 0 is more likely
        sentimentchance = self.score(data)

        if sentimentchance['1'] > sentimentchance['0']:
            return '1' #more likely negative
        
        else:
            return '0' #more likely positive
        
    #create tuple for sentiment and word pairings
    def featurize(self, data): #functional but not used in train - positive word and negative word dictionaries are easy to build in loops
    
        wordSentimentPairs = []
        msg = data[1]
        val = data[2]
        words = msg.split()
        for word in words:
            wordSentimentPairs.append((word, val))
        
        return wordSentimentPairs
        

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:
   
        
            
           
            

    def __init__(self):
    # do whatever you need to do to set up your class here
        self.poscount = 0      
        self.negcount = 0     
        self.posdict = {}
        self.negdict = {}
        self.positivewords = []  
        self.negativewords = []
   
        self.voc = []    

        self.probgivenclass = {}      #P(W|C) formula
        





    #need to get rid of miscellaneous modifiers to words to learn better  
    def dataclean(self, datainp):
        
        newtuple = []
        
        for i in range(len(datainp)):
            #lowercase all data for consistency
            line = datainp[i][1].lower()
            line = re.sub(r'\s+',' ', line)
            line = re.sub('[^a-zA-Z0-9\']', ' ', line) #getting rid of attached punctuation with regex

            if(len(datainp[0]) == 3):
                newtuple.append((datainp[i][0], line, datainp[i][2]))
            else:
                newtuple.append((datainp[i][0], line))
        return newtuple
            

    def train(self, examples):

        nummsg = len(examples)

        for i in range(nummsg):
            
            msg = examples[i][1]
            senti = examples[i][2]
            ##POPULATE TUPLES FOR NEGATIVE WORDS AND POSITIVE WORDS      
            if senti == '1': #keep track of overall pos/neg ratio

                self.positivewords.append(msg)
                self.poscount = self.poscount + 1
            else:
                self.negativewords.append(msg)
                self.negcount = self.negcount + 1

     

        for linepos, lineneg in zip(self.positivewords, self.negativewords):
            wordspos = linepos.split()
            wordsneg = lineneg.split()
            for word in wordspos:
                if word in self.posdict:
                    self.posdict[word] = self.posdict[word] + 1 #add 1 to how many times that word occurs in positive contexts
                else:
                    self.posdict[word] = 1 #or start at 1 because not yet seen
            for word in wordspos:
                if not(word in self.voc):
                    self.voc.append(word)
            for word in wordsneg:
                if word in self.negdict:
                    self.negdict[word] +=1
                else:
                    self.negdict[word] = 1
            for word in wordsneg:
                if not(word in self.voc):
                    self.voc.append(word)
                
            
                
        #get the words and frequency word and store in frequency negative dictionary

                    
 
        self.chancepos = sum(self.posdict.values())
        self.chanceneg = sum(self.negdict.values())
        self.vsize = len(self.voc)
        
        #number of document
        nummsg = len(examples)

        #overall chance of neg or pos msg
        
        self.classchance = {}       #probability of class overall needed in formulas
        self.classchance['1'] = self.poscount/nummsg
        self.classchance['0'] = self.negcount/nummsg
        
        #find probability of word given class - LAPLACE SMOOTHING FORMULA FROM LECTURE
        for word in self.voc:
            if word in self.negdict:

                probbad = (self.negdict[word] + 1) / (self.chanceneg + self.vsize) #laplace again throughout
                self.probgivenclass[(word,'0')] = probbad ##updating probability 


            else:

                probbad = 1 / (self.chanceneg + self.vsize) #if word not found, handled with smoothing
                self.probgivenclass[(word,'0')] = probbad


        
            if word in self.posdict:

                probgood = (self.posdict[word] + 1) / (self.chancepos + self.vsize) #laplace smoothing formula! access dict for amount of occurrences of word
                self.probgivenclass[(word,'1')] = probgood
        

            else: 
                probgood = 1 / (self.chancepos + self.vsize) #this is done if the word is not found to avoid zeroing out
                self.probgivenclass[(word,'1')] = probgood
        
    
    def score(self, data): ##get chance it is positive or negative - calculate P(c)*P(data| c) 
        probgood = 0
        probbad = 0

        words = data.split()

        totalprobpos = 0.0 #sentence classing
        totalprobneg = 0.0
        
        for word in words:
            if word in self.voc: #check if its in the vocab; if it's not, just ignore it
                probgood += np.log(self.probgivenclass[(word,'1')])
                probbad += np.log(self.probgivenclass[(word,'0')])
            
                
        totalprobpos = np.exp(np.log(self.classchance['1']) + probgood)
        totalprobneg = np.exp(np.log(self.classchance['0']) + probbad)
        
        return { '1' : totalprobpos, '0' : totalprobneg}
                
                
    def classify(self, data):

        #create dict from score and see if 1 or 0 is more likely
        sentimentchance = self.score(data)

        if sentimentchance['1'] > sentimentchance['0']:
            return '1' #more likely negative
        
        else:
            return '0' #more likely positive
        
    #create tuple for sentiment and word pairings
    def featurize(self, data): #functional but not used in train - positive word and negative word dictionaries are easy to build in loops
    
        wordSentimentPairs = []
        msg = data[1]
        val = data[2]
        words = msg.split()
        for word in words:
            wordSentimentPairs.append((word, val))
        
        return wordSentimentPairs
        

    def __str__(self):
        return "Cleaner data assessment"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)
    print("starting")

    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    
    # do the things that you need to with your base class
    traintuples = generate_tuples_from_file(training)
    devtest = generate_tuples_from_file("dev_file.txt")
    inputtest = generate_tuples_from_file(testing)
    sa.train(traintuples)
    print("passed training")
    
    predictions = predclass(sa, devtest)
    truelabels = reallabels(devtest)
    print(sa)

    print("Recall of SA: ", recall(truelabels, predictions))
    print("Precision of SA: ", precision(truelabels, predictions))
    print("F1 score of SA: ", f1(truelabels, predictions))

    for row in inputtest:#to get desired output to file
        writetofile("label_test_data.txt", row[0], sa.classify(row[1]))


    


    print("Written to file.")
 


########################################################### IMPROVED STUFF
    
    saimp = SentimentAnalysisImproved()
    print(saimp)


    traintuples2 = generate_tuples_from_file(training)
    trainnew = saimp.dataclean(traintuples2)
    devtestnew = saimp.dataclean(devtest)
    saimp.train(trainnew)
    testnew = saimp.dataclean(inputtest)
    

    
    predictions2 = predclass(saimp, devtestnew)
    

    print("Recall of SA improved: ", recall(truelabels, predictions2))
    print("Precision of SA improved: ", precision(truelabels, predictions2))
    print("F1 score of SA improved: ", f1(truelabels, predictions2))

    for row in testnew:
        writetofile("improved_label_test_data.txt", row[0], saimp.classify(row[1]))
    
    
