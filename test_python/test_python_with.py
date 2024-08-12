import sys
import os


class WithFile:
    # 1. __init__()  初始化方法
    def __init__(self, file_name, file_mode):
        self.file_name = file_name
        self.file_mode = file_mode

    # 2. __enter__() 上文方法
    def __enter__(self):
        print("进入上下文")
        # 将打开的文件名和打开文件的方式赋给self.file
        self.file = open(self.file_name, self.file_mode)
        return self.file

    # 3. __exit__() 下文方法
    # 这三个参数分别代表：异常类型 (exc_type)、值 (exc_value) 及回溯信息 (traceback)
    def __exit__(self, exc_type,exc_val,exc_tb):
        print("退出上下文")
        if self.file:
            self.file.close()



def test():
    print("cur work path:", os.getcwd())
    print("sys path", sys.path)
    with WithFile('hello.txt', 'r') as file:
        file_data = file.read()
        print(file_data)


if __name__ == "__main__":
    #print("args:", sys.argv)
    test()
