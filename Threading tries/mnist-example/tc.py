import socket
import sys
import os
import time
from datetime import datetime

if __name__ == "__main__":

    # parse parameters
    if (len(sys.argv) == 2):
        modeltype = str(sys.argv[1])
    else:
        print 'not enough input arguments'
        exit()
    params = modeltype + '\n\r'

    # create client socket to server
    host = socket.gethostname()
    port = 8999
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print 'connecting'
        s.connect((host,port))
    except:
        print 'error connecting with background process'
        exit()

    # send modeltype to server
    try:
        s.sendall(params)
    except:
        print 'sendall error'

    # get results back from server and print them in terminal
    while True:
        # this is non blocking socket so use exceptions to handle case when no data is received
        try:
            new_data=s.recv(1024)
            print new_data
            if new_data[len(new_data)-1] == '\r':
                break
        except:
            continue

    s.close()
