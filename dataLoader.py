from colorama import Fore
from numpy import *

class Corpus():

    def __init__(self, txt):
        """
        Interface:

            self.X ->
            self.Y -> 
            self.numToWord ->
            self.wordToNum -> 
        """

        # Assign Variables
        self.text = txt
        self.vocab = set(["UNKNOWN","<s>", "</s>"])
        self.numToWord = {}
        self.wordToNum = {}
        self.brown = False
        
        # Process data
        self.getSentances()
        self.generateVocabulary()
        self.seqs_to_XY()

    def brownInit(self, num=200, offset=0):
        
        from nltk.corpus import brown
        self.brown = True

        self.sentances = brown.sents()[offset:(offset+num)]

        self.vocab = set(["UNKNOWN","<s>", "</s>"])
        for s in self.sentances :
            for w in s :
                self.vocab.add(w.lower())


        self.V = len(self.vocab)        
        self.numToWord = dict(enumerate(self.vocab))
        # self.wordToNum = {v:k for k,v in self.numToWord.iteritems()}
        self.wordToNum = dict((v,k) for k,v in self.numToWord.iteritems())

        self.seqs_to_XY()
    
    def getSentances(self):
        self.sentances = self.text.split(".")
        return self.sentances

    def generateVocabulary(self):

        for w in self.text.split(' '):
            self.vocab.add(w.lower())

        self.V = len(self.vocab)        
        self.numToWord = dict(enumerate(self.vocab))
        # self.wordToNum = {v:k for k,v in self.numToWord.iteritems()}
        self.wordToNum = dict((v,k) for k,v in self.numToWord.iteritems())

        # print Fore.GREEN, self.wordToNum['on']

    def seq_to_indices(self, words):

        return array([self.wordToNum.get(w.lower(), 0) for w in words])

    def docs_to_indices(self, docs=None):
        # docs = [pad_sequence(seq, left=1, right=1) for seq in docs]
        if docs == None:
            docs = self.sentances
        ret = []
        for seq in docs:
            # words = [canonicalize_word(wt[0], word_to_num) for wt in seq]
            try:
                words = seq.split(' ')
            except:
                words = seq

            words.append("</s>")
            words.insert(0,"<s>")
            if len(words) < 4:
                continue
            ret.append(self.seq_to_indices(words))

        # return as numpy array for fancier slicing
        return array(ret, dtype=object)

    def offset_seq(self, seq):
        return seq[:-1], seq[1:]

    def seqs_to_XY(self):

        seqs_with_idx = self.docs_to_indices(self.sentances)

        X, Y = zip(*[self.offset_seq(s) for s in seqs_with_idx])
        self.X, self.Y = array(X, dtype=object), array(Y, dtype=object)


    def idx_to_sentance(self, seq):

        out = [self.numToWord.get(s, "_") for s in seq]
        return ' '.join(out)

    def get_XY_For_another_corpus(self, c):

        seqs_with_idx = self.docs_to_indices(c.sentances)

        X, Y = zip(*[self.offset_seq(s) for s in seqs_with_idx])
        X, Y = array(X, dtype=object), array(Y, dtype=object)
        return X, Y


