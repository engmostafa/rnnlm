import unittest
import Queue
from ui import UIThread
import time

class TestUI(unittest.TestCase):

    def test_UIThread(self):

        print "Testing...."
    
        inp_q = Queue.Queue()
        out_q = Queue.Queue()
        ui = UIThread(inputChannel=inp_q, sigChannel=out_q)
        ui.start()

        for i in range(6):
            time.sleep(1)
            msg = {'Speed': float(i), 'Distance': 20*i}
            inp_q.put(msg)

        ui.exitSelf()
