from rnn import RNN
from colorama import Fore
import sys
import numpy as np

class Trainer():

    def __init__(self, c, rnn, nepochs = 12, alpha = 0.1):

        self.nepochs = nepochs
        self.alpha = alpha
        self.c = c
        self.rnn = rnn


    def train(self):
        
        X = self.c.X
        Y = self.c.Y

        print Fore.BLUE, "\n Started Training \n"
        ecosts = []
        for e in xrange(self.nepochs):
            lastCost = 0.0 
            for i in range(len(X)):
            # for i in range(100):

                xseq = X[i]
                yseq = Y[i]

                if i < 3:
                    niter = 4 - i
                else:
                    niter = 1
        
                for k in xrange(niter):
                    try:
                        cost = self.rnn.train(list(xseq), list(yseq), self.alpha) , 
                        print Fore.RED, ("Sent #%d, Epoch #%d Cost=%f  \r")%(i, e, cost[0]),
                        sys.stdout.flush()

                    except KeyboardInterrupt:
                        self.rnn.save()
                        self.generate_sequence()
                        # print Fore.BLUE, "\nACC = ", self.fullDataTest()


                lastCost += cost[0]
                avgCost = lastCost / (i+1)

                # print Fore.YELLOW,
                # print (" TotAvgCost = %f at epoch %d, sentance %d \r")%(avgCost, e+1, i),
                # sys.stdout.flush()

            # print Fore.GREEN, "AvgCost for epoch #%d = %f  \n\r"%(e+1, avgCost)
            ecosts.append(avgCost)

            self.rnn.save(ep = e)
            print Fore.BLUE, "\nACC = ", self.fullDataTest()


    def fullDataTest(self):

        X = self.c.X
        Y = self.c.Y
        
        totAcc = 0.0
        for i in range(len(X)):
            xseq = X[i]
            yseq = Y[i]

            acc = self.test_sentance(xseq, yseq)
            totAcc += acc
            avgAcc = totAcc / (i+1)
            
            print Fore.GREEN , "#%d - Average Acc %.2f %% \r" % (i, avgAcc),
            sys.stdout.flush()


        return totAcc/len(X)

    def test_sentance(self,x , y):
        
        ypred = self.rnn.classify(list(x),list(y))
        
        # print Fore.GREEN, self.c.idx_to_sentance(x)
        # print Fore.GREEN, self.c.idx_to_sentance(y)
        # print Fore.CYAN, self.c.idx_to_sentance(list(ypred))
        
        accuracy = float(np.count_nonzero(ypred-y))
        accuracy /= float(len(y))

        accuracy = 100*(1.0-accuracy)
        # print Fore.CYAN, ypred-y 
        # print Fore.GREEN, accuracy, "%\n"

        return accuracy


    def generate_sequence(self):

        seq = [self.c.wordToNum["<s>"]] 

        maxLen = 20

        for i in range(maxLen):

            dummyY = range(len(seq))
            ypred = self.rnn.classify(seq, dummyY)
            lastWordIdx = list(ypred)[-1]
            seq.append(lastWordIdx)

        print Fore.GREEN, self.c.idx_to_sentance(seq)

