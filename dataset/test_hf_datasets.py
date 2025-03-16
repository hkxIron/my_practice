from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets
from datasets.formatting.formatting import LazyRow
import pandas as pd
import sys

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

def test_ds2():
    import numpy as np
    from datasets import Dataset

    seq_len, dataset_size = 512, 512
    dummy_data = {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
        "labels": np.random.randint(0, 1, (dataset_size)),
    }
    ds = Dataset.from_dict(dummy_data)
    ds.set_format("pt")
    print('\n')
    print("pandas data:",ds[:3])

def test_init_empty():
    import torch.nn as nn
    from accelerate import init_empty_weights

    """
    # 测试好像不行，仍然会分配内存
    with init_empty_weights():
        model = nn.Sequential([nn.Linear(100000, 100000) for _ in range(1000)])  # This will take ~0 RAM!
    """

def test_gsm8k():
    data_dir="/home/hkx/data/work/hf_data_and_model/datas/openai/gsm8k/main"
    #data = load_dataset(path=data_dir, name='main')[split]
    data = load_dataset(path="parquet", data_files={
        "train":"/home/hkx/data/work/hf_data_and_model/datas/openai/gsm8k/main/train-*.parquet",
        "test":"/home/hkx/data/work/hf_data_and_model/datas/openai/gsm8k/main/test-*.parquet",
    })
    print(data)
    print(data["train"])
    df = data["train"].to_pandas()
    df.to_csv("data/gsm8k/train.csv",index=None)
    # rbql;  select a1 where a1.indexOf("James")!=-1
    # https://github.com/mechatroner/vscode_rainbow_csv/blob/master/rbql_core/README.md#rbql-rainbow-query-language-description
    data["test"].to_pandas().to_csv("data/gsm8k/test.csv",index=None)


def test_glue():
    data = load_dataset("nyu-mll/glue", "mrpc")
    print(data)
    print(data["train"])
    df = data["train"].to_pandas()
    df.to_csv("data/glue/train.csv",index=None)


if __name__ == "__main__":
    print("sys args:")
    print(sys.argv)

    test_glue()
    sys.exit()
    test_gsm8k()
    test_ds()
    test_ds2()
    test_init_empty()
