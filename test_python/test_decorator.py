import functools
import time
from typing import Callable

from termcolor import cprint


def my_decorator(func:Callable):
    def my_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} run time:{end_time-start_time}")
        return result
    return my_wrapper

def square(x):
    return x**2

@my_decorator
def square2(x):
    return x**2


"""
装饰器生成器
"""
def my_timer(threshold:float):
    def my_decorator_inner(func:Callable):
        @functools.wraps(func) # 可以继承func的名称
        def my_wrapper_inner(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if end_time-start_time>threshold:
                print(f"{func.__name__} run time:{end_time-start_time} > {threshold}")
            else:
                print(f"{func.__name__} run time:{end_time - start_time} < {threshold}")
            return result
        return my_wrapper_inner
    return my_decorator_inner

@my_timer(0.2)
def my_sleep():
    time.sleep(0.4)

if __name__ == "__main__":
    """
    下面两种方法等价
    """
    decorator_square = my_decorator(square)
    # 法一
    print(decorator_square(10))
    # 法二
    print(square2(10)) # 比不用装饰器时间更短

    # 装饰器生成器
    my_sleep()
    # 等价于
    my_decorator2 = my_timer(0.2)(my_sleep)
    print(my_decorator2.__name__) #
    my_decorator2()

    cprint(f"这是红色", "red")

