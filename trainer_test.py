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

    def test_generating_sentance(self):

        self.trainer.generate_sequence()

    def tesT_saving_model(self):
        c = Corpus(self.txt)
        rnn = RNN(100, c.V, 50)
        rnn.save()

