from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets
from datasets.formatting.formatting import LazyRow
import pandas as pd

def add_sum_col(row:LazyRow):
    row["sum"] = row["a"]+row["b"]
    row["a2"] = row["a"]*2
    return row

def test_ds():
    my_dict = {"a": range(0, 20, 1), "b":range(-3, 17, 1)} # 每个key对应一列
    dataset = Dataset.from_dict(my_dict)  # 从dataFrame导入数据
    print("dataset:", dataset)

    print("部分数据:", dataset[0]) # 直接打印数据
    print("部分数据:", dataset[0:5]) # 直接打印数据
    print("部分数据:", dataset.select(range(5))[0:5]) # select 会新生成一个数据集

    # 变换
    ds2 = dataset.map(function=add_sum_col)
    print("map:", ds2[0:5])

    # 过滤
    ds2 = ds2.filter(lambda x:x["sum"]>1)
    print("filter:", ds2[0:5]) # 直接打印数据

    # 排序
    ds2 = ds2.sort(column_names=['sum'], reverse=True)
    print("sort:", ds2[0:5]) # 直接打印数据


    print("take 2th data:", ds2.take(3))
    print("columns:", ds2.column_names)
    print("columns keys", list(ds2.take(1))[0].keys())

    # 转换成dataframe
    df = ds2.select(range(5))
    df.set_format("pandas")
    print('\n')
    print("pandas data:", df[:3])

    # 从dataframe中初始化
    df = pd.DataFrame({"a": [1, 2, 3], "b":[4,5,6]})
    dataset2 = Dataset.from_pandas(df)
    print(dataset2)  # 查看数据的结构


if __name__ == "__main__":
    test_ds()
