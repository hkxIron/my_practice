import sys
import os
def show_paths():
    print("cur work path:", os.getcwd())
    print("sys path", sys.path)


if __name__ == "__main__":
    #print("args:", sys.argv)
    show_paths()
