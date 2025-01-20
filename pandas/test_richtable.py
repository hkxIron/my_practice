import pandas as pd
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table

def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    print("origin pandas:")
    print(df)

    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)

def test1():
    title="数据表"
    df = pd.DataFrame({
        "name":["张三", "李四", "王五"],
        "salary":[3000, 4000, 5000]
    })
    console = Console(force_terminal=True)
    print_rich_table(title, df, console)


if __name__ == '__main__':
    test1()