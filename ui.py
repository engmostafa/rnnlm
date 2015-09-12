import curses
import time
import threading, Queue
import curses.textpad as textpad

class UIThread(threading.Thread):
    
    def __init__(self, inputChannel, sigChannel = None):
        super(UIThread, self).__init__()

        self.stoprequest = threading.Event()

        self.myscreen = curses.initscr()
        curses.start_color()

        self.myscreen.border(0)
        self.myscreen.timeout(0)
        curses.noecho()
        curses.cbreak()
        # self.myscreen.keypad(1);
        self.textbox = textpad.Textbox(self.myscreen)

        self.Lastinfo = []


        curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_CYAN)

        self.inp_q = inputChannel
        self.sig_q = sigChannel

    def getTextString(self):
        
        win = curses.newwin(5, 60, 5, 40)
        tb = textpad.Textbox(win)
        strng = tb.edit()
        return strng

    def run(self):
        # As long as we weren't asked to stop, try to take new tasks from the
        # queue. The tasks are taken with a blocking 'get', so no CPU
        # cycles are wasted while waiting.
        # Also, 'get' is given a timeout, so stoprequest is always checked,
        # even if there's nothing in the queue.

        while not self.stoprequest.isSet():
            try:
                info = self.inp_q.get(True, 0.05)
                self.updateInfo(info)

            except Queue.Empty:

                ch = self.myscreen.getch()
                
                if ch == -1 :
                    continue

                self.sig_q.put(ch)

                if ch == ord('q'):
                    self.exitSelf()

                

    def updateInfo(self, info):

        self.curYPos = 2 
        if info.keys() != self.Lastinfo :
            self.myscreen.clear()

        for k in info :
            
            strPrnt = "%s: %s"%(k, info[k])
            self.myscreen.addstr(self.curYPos, 5, strPrnt )
            self.curYPos += 2
        
        self.myscreen.refresh()
        self.LastInfo = info.keys()


    def exitSelf(self):
        self.stoprequest.set()
        curses.endwin()
        curses.nocbreak() 
        # self.myscreen.keypad(0)
        curses.echo()

        

    def join(self, timeout=None):
        super(UIThread, self).join(timeout)
        self.exitSelf()


