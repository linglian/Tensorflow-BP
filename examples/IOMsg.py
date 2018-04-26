from multiprocessing.connection import Listener
from multiprocessing.connection import Client
import traceback

def echo_client(conn):
    try:
        while True:
            msg = conn.recv()
            conn.send(msg)
    except EOFError:
        print 'Connection closed'

def echo_server(address, authkey):
    serv = Listener(address, authkey=authkey)
    while True:
         try:
             client = serv.accept()
             echo_client(client)
         except Exception:
            traceback.print_exc()



if __name__ == '__main__':
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'sc')
    for op, value in opts:
        if op == '-s':
            echo_server('./temp2', authkey=b'peekboo')
        elif op == '-c':
            c = Client('./temp2', authkey=b'peekboo')
            c.send('hello')
            print c.recv()
            c.send(42)
            print c.recv()
            c.send([123,2,2,2312,321,88])
            print c.recv()



