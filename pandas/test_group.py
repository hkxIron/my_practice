import pandas as pd

"""
测试pandas group

1.apply中如何将一行数据变成多行

df = pd.DataFrame({'class': ['A', 'B', 'C'], 'count':[1,0,2]})

  class  count
0     A      1
1     B      0
2     C      2

转换为如下数据
  class 
0     A   
1     C   
2     C 
"""


def apply_return_multi_row():
    df = pd.DataFrame({'class': ['A', 'A', 'B', 'C'],
                       'count': [1, 2, 0, 2]}
                      )
    print(df)

    def f(group):
        print(type(group)) # pd.DataFrame
        print("group:\n", group)
        first_row = group.iloc[0] # 取dataFrame的第一行
        return pd.DataFrame({'class': [first_row['class']] * first_row['count']})

    df2 = df.groupby('class', group_keys=False).apply(f)
    #df2 = df.groupby('class', group_keys=True).apply(f)
    print(df2)


if __name__ == '__main__':
    apply_return_multi_row()
