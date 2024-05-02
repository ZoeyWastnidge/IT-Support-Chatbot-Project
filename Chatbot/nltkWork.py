import nltk
import numpy as np
#nltk.download('punkt')                      #Downloads a pre trained tokeniser algorithm (splits up sentences into smaller sentences)  - only used once
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenise(sentence):
    return nltk.word_tokenize(sentence)     #Uses punkt to split up sentences into tokens

def stem(word):
    return stemmer.stem(word.lower())       #Finds the base form of words through stemming

def bagOfWords(tokenSentence, allWords):    #Generates the bag of words
    

    tokenSentence = [stem(w) for w in tokenSentence]

    bag = np.zeros(len(allWords), dtype=np.float32)
    for id, w in enumerate(allWords):
        if w in tokenSentence:
            bag[id] = 1.0
    
    return bag

