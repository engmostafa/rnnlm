from rnn import RNN
from colorama import Fore
import sys
import numpy as np
import Queue
from ui import UIThread

class Trainer():

    def __init__(self, c, rnn, nepochs = 12, alpha = 0.1):

        self.nepochs = nepochs
        self.alpha = alpha
        self.c = c
        self.rnn = rnn

        self.display_q = Queue.Queue()
        self.signal_q = Queue.Queue()
        self.ui = UIThread(inputChannel=self.display_q,sigChannel=self.signal_q)
        self.ui.start()

        info_st = {
                "p": "Start/Pause Training",
                "y": "Save Params",
                "s": "Sentance Testing",
                }
        self.display_q.put(info_st)

        self.scores = {}

    def sendMsg(self, msg):

        info = {"msg": msg}
        self.display_q.put(info)

    def mainEventLoop(self):

        sig = 0

        while (sig != 1 and sig !=2):
            sig = self.checkForSignals()
            if sig == 2:
                self.sendMsg("Training Called")
                self.train()




    def train(self):
        
        self.sendMsg("Starting Training")
        X = self.c.X
        Y = self.c.Y

        self.lastAvg = 0.0

        ecosts = []
        for e in xrange(self.nepochs):
            lastCost = 0.0 
            for i in range(len(X)):

                xseq = X[i]
                yseq = Y[i]

                if i < 3:
                    niter = 4 - i
                else:
                    niter = 1
        
                for k in xrange(niter):
                    try:
                        cost = self.rnn.train(list(xseq), list(yseq), self.alpha)

                        info = {
                                'Sentance': i,
                                'Epoch': e,
                                'Cost': cost,
                                'Last Average': self.lastAvg,
                                'Scores': self.scores,
                        }
                        self.display_q.put(info)
                        sig = self.checkForSignals()

                        if sig == 1:
                            self.quitTraining()
                            return

                        if sig == 2:
                            sig2 = 0
                            while sig2 != 2 :
                                sig2 = self.checkForSignals()


                    except KeyboardInterrupt:
                        pass
                        self.quitTraining()
                        return


                lastCost += cost
                avgCost = lastCost / (i+1)


            ecosts.append(avgCost)

            self.rnn.save("epoch%d"%e)
            # self.fullDataTest()


    def quitTraining(self):

        self.rnn.save("beforequit")
        print Fore.RED, "\n\n QUITTING TRAINING...\n\n"

    def checkForSignals(self):

        try:
            msgCh = self.signal_q.get_nowait()
            
            if msgCh == ord('q'):
                return 1
            
            if msgCh == ord('p'):
                return 2

            if msgCh == ord('s'):
                strng = self.ui.getTextString()
                strng = strng.replace("\n", "")
                print strng.split(' ')
                score = self.findProbabilityOfSentance(strng)
                self.scores[strng] = score
                self.display_q.put(self.scores)

            if msgCh == ord('t'):
                self.calcPerplexicity()

            if msgCh == ord('c'):
                self.fullDataTest()

            if msgCh == ord('y'):
                strng = self.ui.getTextString()
                strng = strng.replace("\n", "")
                self.rnn.save("%s.save"%strng)


        except Queue.Empty :
            pass

        return 0


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
            
            info = {
                    'Sentance': i,
                    'AvgCost': "%.2f %%"%avgAcc,
            }
            self.display_q.put(info)
            self.lastAvg = avgAcc


        return totAcc/len(X)

    def test_sentance(self,x , y):
        
        ypred = self.rnn.classify(list(x),list(y))
        
        
        accuracy = float(np.count_nonzero(ypred-y))
        accuracy /= float(len(y))

        accuracy = 100*(1.0-accuracy)

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


    def calcPerplexicity(self, c=None):

        if c == None:

            X = self.c.X
            Y = self.c.Y
        else :
            X,Y = self.c.get_XY_For_another_corpus(c)
        
        totAcc = 0.0
        totWords = 0
        totPer = 0.0
        minPer = 999999
        maxPer = 0
        for i in range(len(X)):
            xseq = X[i]
            yseq = Y[i]

            # curP = self.Perplexicity_xy(xseq, yseq)

            cost = self.rnn.cost(list(xseq),list(yseq))
            curP =  np.exp(cost)
            totPer += curP
            if maxPer < curP:
                maxPer = curP
            if minPer > curP and curP > 10:
                minPer = curP
            totAcc += cost * len(xseq)
            totWords += len(xseq)

            avgAcc = np.exp(totAcc /totWords )
            totSumAvg = totPer / (i+1)

            info = {
                    'Sentance': i,
                    'PERPLEXITY': "%.2f"%avgAcc,
                    'Current': "%.2f"%curP,
                    'Min': minPer,
                    'Max': maxPer,
                    'SumAvgPer' : totSumAvg,
            }
            self.display_q.put(info)

        return avgAcc


    def Perplexicity_xy(self,x,y):

        cost = self.rnn.cost(list(x),list(y))
        return np.exp2(cost)

    def findProbabilityOfSentance(self, sentance = "The president said he will ask congress"):

        words = sentance.split(' ')

        # words.append("</s>")
        # words.insert(0,"<s>")
        xseq = self.c.seq_to_indices(words)
        y = xseq[1:] #= range(len(xseq))
        x = xseq[:-1]
        
        return self.Perplexicity_xy(x,y)
