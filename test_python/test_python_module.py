import sys
import os
def show_paths():
    print("cur work path:", os.getcwd())
    print("sys path", sys.path)


if __name__ == "__main__":
    #print("args:", sys.argv)
    show_paths()


"""
在pycharm中，“Mark Directory as Sources Root” 可以将选中的文件夹添加到sys.path中
会改变当前环境的PYTHONPATH

在python中等价于：
import sys
sys.path.insert(0, my_folder)

而在shell中等价于：
export PYTHONPATH="${PYTHONPATH}:/your/source/root"

"""