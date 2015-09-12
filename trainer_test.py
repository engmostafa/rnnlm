import unittest
from colorama import Fore
from rnn import *
from dataLoader import Corpus
from trainer import Trainer

class TestRNNTrainer(unittest.TestCase):

    def setUp(self):

        fname = 'friends.txt'
        f = open(fname, 'r')
        self.txt = f.read()
        f.close()


        c = Corpus(self.txt)
        c.brownInit(500)
        rnn = RNN(100, c.V, 50)
        rnn.load('rnn.save')

        self.trainer = Trainer(c,rnn, nepochs=50, alpha = 1.8)

    def tesT_TrainingOnSentances(self):

        c = Corpus(self.txt)
        rnn = RNN(100, c.V, 50)

        trainer = Trainer(c,rnn, nepochs=50, alpha = 1.8)
        trainer.train()

    def tesT_SentancePrintTest(self):

        # trainer.test_sentance(c.X[1], c.Y[1])
        self.trainer.fullDataTest()

    def test_perplexicity(self):

        p = self.trainer.calcPerplexicity()
        print Fore.CYAN, p, "\n"

    def tesT_evaluating_sentance(self):
        
        sentances = [
            "I love my son",
            "This food makes no sense",
            "The president loves to eat food",
            "love stock bullet is are am",
            "sense I of am since is",
            "the number of voters and the size of this city",
            "considering the widespread election",
            "implementation of the law",
            "on horse race betting",
            "He is willing to stake his political career on it",
            "He is willing to stake her political career on it",
            "He willing to stake her political career on it",
            "He is willing to stake political career it",
            "I'm willing to stake my political career on it",
            "I'm willing to stake my political career on this",
            "all the voters are willing to stake their political career on it",
            "all the voters am willing to stake my political career on it",
        
        ]

        for s in sentances:

            p = self.trainer.findProbabilityOfSentance(s)
            print Fore.CYAN, "%s -> %f\n\r" % (s,p)

    def tesT_generating_sentance(self):

        self.trainer.generate_sequence()

    def tesT_saving_model(self):
        c = Corpus(self.txt)
        rnn = RNN(100, c.V, 50)
        rnn.save()

