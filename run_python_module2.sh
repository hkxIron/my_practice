#python3 -m my_practice/test_python/test_python_module.py

:<<EOF
cur work path: /media/hkx/win/hkx/ubuntu/work/open/my_practice
sys path ['/media/hkx/win/hkx/ubuntu/work/open/my_practice/test_python', '/home/hkx/miniconda3/lib/python311.zip', '/home/hkx/miniconda3/lib/python3.11', '/home/hkx/miniconda3/lib/python3.11/lib-dynload', '/home/hkx/miniconda3/lib/python3.11/site-packages']
EOF
python3 test_python/test_python_module.py # 不带-m会将test_python_module.py所在目录加入sys.path

:<<EOF
cur work path: /media/hkx/win/hkx/ubuntu/work/open/my_practice
sys path ['/media/hkx/win/hkx/ubuntu/work/open/my_practice', '/home/hkx/miniconda3/lib/python311.zip', '/home/hkx/miniconda3/lib/python3.11', '/home/hkx/miniconda3/lib/python3.11/lib-dynload', '/home/hkx/miniconda3/lib/python3.11/site-packages']
EOF
python3 -m test_python.test_python_module # 带-m会将当前工作目录加入sys.path
