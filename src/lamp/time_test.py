#!/usr/bin/env python3
'''Time Test Tools'''
from time import time
s=time()
i = 0
while i < 1000000000:
    i += 1
e=time()
print(e-s)
