import os
import time
from datetime import datetime

def show_time():
    dt = datetime.now()
    print("当前时间", dt.strftime('%Y.%m.%d %H:%M:%S'))
