from rnn import *
from dataLoader import Corpus
from trainer import Trainer

fname = 'shakespear.txt'
f = open(fname, 'r')

txt = f.read()

f.close()

c = Corpus(txt)
c.brownInit(500)

rnn = RNN(100, c.V, 50)
# rnn = RNN.load('rnn.save')
rnn.load('rnn.save')

# rnn = RNN(100, c.V, 50)

trainer = Trainer(c,rnn, nepochs=50, alpha = 0.001)
trainer.generate_sequence()
# trainer.train()
trainer.mainEventLoop()
