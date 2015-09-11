import unittest
from colorama import Fore
from rnn import *

class TestRNN(unittest.TestCase):

    def test_RNN_Train(self):
        
        rnn = RNN(10, 40, 50)
        
        test_x = [1,2,3]
        test_y = [1,2,3]

        print Fore.CYAN, rnn.cost(test_x, test_y)
        for i in range(100):
            print Fore.GREEN, rnn.train(test_x, test_y, 0.5)
