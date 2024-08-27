from dataclasses import dataclass
from collections import namedtuple
from typing import *

"""
快速写出简化类
"""
@dataclass
class Vec:
    x:int
    y:int

Location = namedtuple("Location", ["x", "y"])

def main():
    v = Vec(2,2)
    print(v)
    v.x=10 # 这个可以改
    print(v)

    loc = Location(5, 10)
    #loc.x=10 # 这个不能改
    print(loc)
    print(loc.x)




main()