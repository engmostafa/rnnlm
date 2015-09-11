import unittest
from colorama import Fore
from dataLoader import *

class TestDataLoader(unittest.TestCase):

    def setUp(self):

        fname = 'shakespear.txt'
        f = open(fname, 'r')
        self.txt = f.read()
        f.close()

    def Test_loading_sentences(self):

        sentencesList = getSentances(self.txt)

        print Fore.GREEN, sentencesList[:2]


    def Test_loading_XY_Pair(self):

        c = Corpus(self.txt)
        X = c.X

        print Fore.CYAN , X[0]
        print Fore.GREEN , c.Y[0]

