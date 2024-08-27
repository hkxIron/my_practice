import functools
import time
from typing import Callable

from termcolor import cprint
from functools import *
from itertools import *
from operator import *

def my_decorator(func:Callable):
    def my_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} run time:{end_time-start_time}")
        return result
    return my_wrapper


def test1():
    n=6
    print(reduce(lambda a,b: a*b, range(1,n)))
    print(reduce(lambda a,b: a*b, list(range(1,n))))
    print(eval("*".join([str(i) for i in list(range(1,n))])))
    #print(reduce(mul, range(1,6)))
    print(reduce(mul, range(1,n)))

    # 仅针对相邻的进行groupby
    x=groupby("aabbccccdddaa")
    print(list(x))

    print([k for k,g in groupby("AAABBCCAA")])
    # ['A', 'B', 'C', 'A']

    print([(k,list(g)) for k,g in groupby("AAABBCCAA")])
    # [('A', ['A', 'A', 'A']), ('B', ['B', 'B']), ('C', ['C', 'C']), ('A', ['A', 'A'])]

    print([(k,list(g)) for k,g in groupby(range(0, 10), lambda x:x%3==0)])
    # [(True, [0]), (False, [1, 2]), (True, [3]), (False, [4, 5]), (True, [6]), (False, [7, 8]), (True, [9])]

    #a,b = map(int, input().split())
    #print()
    a,b = map(int, ["15", "20"])

    l1 = list(map(str.upper, ['aaa', "bbb"]))
    print(l1)

    # accumulate
    print(list(accumulate(range(10))))

    # filter
    print(list(filter(lambda x:x%3==0, range(10))))
    print(list(filter(lambda x:not x%3, range(10))))
    print(list(filter(lambda x:x%3!=0, range(10))))

    #compress
    print(list(compress("ABCD", [True, False, True, True])))

    # dropwhile
    print(list(dropwhile(lambda x:x<5, range(10))))
    print(list(takewhile(lambda x:x<5, range(10))))

    # islice
    g = (i for i in range(10))
    for x in islice(g, 5):
        print(x)
    # 0
    # 1
    # 2
    # 3
    # 4

    print(list(chain("abc", "def")))
    # ['a', 'b', 'c', 'd', 'e', 'f']

    print(list(chain("abc", "def", [1,2,3])))
    # ['a', 'b', 'c', 'd', 'e', 'f', 1, 2, 3]

    print(list(chain.from_iterable(["abc", "BCD", "QWE"])))
    #['a', 'b', 'c', 'B', 'C', 'D', 'Q', 'W', 'E']



if __name__ == "__main__":
    test1()

