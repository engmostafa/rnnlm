from colorama import Fore
from numpy import *
class Corpus():

    def __init__(self, txt):

        # Assign Variables
        self.text = txt
        self.vocab = set([])
        self.numToWord = {}
        self.wordToNum = {}
        
        # Process data
        self.getSentances()
        self.generateVocabulary()
        self.seqs_to_XY()

    def getSentances(self):
        self.sentances = self.text.split(".")
        return self.sentances

    def generateVocabulary(self):

        for w in self.text.split(' '):
            self.vocab.add(w)

        
        self.numToWord = dict(enumerate(self.vocab))
        self.wordToNum = {v:k for k,v in self.numToWord.iteritems()}

        print Fore.GREEN, self.wordToNum['on']

    def seq_to_indices(self, words):

        return array([self.wordToNum.get(w, 0) for w in words])

    def docs_to_indices(self):
        # docs = [pad_sequence(seq, left=1, right=1) for seq in docs]
        docs = self.sentances
        ret = []
        for seq in docs:
            # words = [canonicalize_word(wt[0], word_to_num) for wt in seq]
            words = seq.split(' ')
            ret.append(self.seq_to_indices(words))

        # return as numpy array for fancier slicing
        return array(ret, dtype=object)

    def offset_seq(self, seq):
        return seq[:-1], seq[1:]

    def seqs_to_XY(self):

        seqs_with_idx = self.docs_to_indices()

        X, Y = zip(*[self.offset_seq(s) for s in seqs_with_idx])
        self.X, self.Y = array(X, dtype=object), array(Y, dtype=object)

