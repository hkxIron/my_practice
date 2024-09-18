import argparse

def parser():
    parser = argparse.ArgumentParser(description="训练DaliNet模型")
    parser.add_argument("-w","--weight",
                                        type=int,
                                        choices=[0, 1, 2, 3, 4],
                                        #action="store_true", # store与choices是互斥的
                                        default=0,
                                        help="存放weight的位置")
    parser.add_argument("--use_lora",
                        type=bool,
                        #action="store_false", # 有action就不能有type
                        default=False,
                        help="是否使用lora")
    parser.add_argument("--use_lora2",
                        type=bool,
                        default=False,
                        help="是否使用lora")
    parser.add_argument("--type",
                        type=str,
                        choices=['llama', 'bert'],
                        #action="store_bert",
                        default='llama',
                        help="模型类型")
    parser.add_argument("--dataname",
                        type=str,
                        choices=['llama_data', 'bert_data'],
                        default='llama_data',
                        help="数据类型")
    """
    python test_args1.py --dataname=bert_data --type=bert --weight=1 --use_lora2=True
    Namespace(weight=1, use_lora=False, use_lora2=True, type='bert', dataname='bert_data')
    """
    # 和parse_args()很像，但parse_known_args()在接受命令行多余参数时不会报错
    args = parser.parse_args()
    print(args)

if __name__ == '__main__':
    parser()