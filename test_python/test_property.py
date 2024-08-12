import datetime
import os
class User:
    name = ""
    uid = 1
    # 1. __init__()  初始化方法
    def __init__(self, file_name,  _name="", _id=0):
        self.file_name = file_name
        self.name = _name
        self.uid = _id

    @property
    def id(self)->int:
        return self.uid

    @id.setter
    def set_name_id(self, _id:int): # user.set_name_id=1000
        self.uid = _id
        #return self

    @staticmethod
    def show():
        print("show current time", datetime.datetime.now())


if __name__ == "__main__":
    user = User("test.txt", "hkx", 1)
    print(user.id)
    #user.set_name_id(500) # 报错
    user.set_name_id=1000 # 设置id
    print(user.id)
    User.show() # 静态函数
