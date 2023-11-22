from transformers import AutoModelForCausalLM
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store_true', help='Enable to prepare for model')
parser.add_argument('--model_name', default="facebook/opt-1.3b", type=str, help='add the model name you want to prepare for deepspeedchat')
parser.add_argument('--data', action='store_true', help='Enable to prepare for data')
parser.add_argument('--data_name', default="Dahoas/rm-static", type=str, help='add the dataset name you want to prepare for deepspeedchat')
parser.add_argument('--data_dir', default="../dataset", type=str, help='add the path of dataset you want to save')
args = parser.parse_args()

if args.model:
    print("prepare for model that you want to train for deepspeed chat")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, fast_tokenizer=True)
    model_config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, from_tf=bool(".ckpt" in args.model_name), config=model_config)
    print("model prepare is done")

if args.data:
    print("download the dataset that you want to train for deepspeed chat")
    dataset0 = load_dataset(args.data_name)
    data_dir = args.data_dir + "/" + args.data_name
    dataset0.save_to_disk(data_dir)
