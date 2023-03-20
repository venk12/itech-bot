import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# s = '''Good muffins cost $3.88\nin New York.  Please buy me'''
stemmer = PorterStemmer()
stop_words = ['?','.',',','!']

def tokenize(sentence):
    return nltk.word_tokenize(sentence) 

def stem(word):
    return stemmer.stem(word.lower())

def bagOfWords(tokenized_sentence, all_words):
    # This function returns the list of bog model
    # Inputs:
        # tokenized_sentence: list of tokens based on text input/prompt
        # all_words: list of unique tokens based on all the words available (glossary)
    # Example of the implementation:
    """
    (input 1) tokenized_sentence   = ['hello','how','are','you']
    (input 2) all_words            = ['hi', 'hello', 'I', 'you', 'know', 'bye', 'thank']
    (output)  bog                  = [ 0  ,    1   ,  0 ,   1  ,   0   ,   0  ,    0   ]
    """
    
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
    

def removeStopWords(tokens):
    cleaned = [x for x in tokens if x not in stop_words]
    return cleaned

