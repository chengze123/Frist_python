from re import *
from urllib3 import *
import os
import hashlib

def read_writeline():
    ms=open("D:\师门\CSRtxt\万科A2017年CSR.txt",encoding='utf-8')
    for line in ms.readlines():
        with open("D:\师门\CSRtxt\万科A2017年CSR.doc","a",encoding='utf-8') as mon:
            mon.write(line)

read_writeline()


