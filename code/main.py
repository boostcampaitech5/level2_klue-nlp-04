import argparse

from train import *
from inference import *
import configparser

def main(args, config=None):
    if args.mode == "train" :
        print("Training mode")
        train(args, config)
        
    elif args.mode == "inference" :
        print("Inference mode")
        main_inference(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--model_name', type=str, default="klue/bert-base")
    parser.add_argument('--config_path', type=str, default="")

    parser.add_argument('--model_path', type=str, default="./best_model")
    parser.add_argument('--tokenizer_name', type=str, default="klue/bert-base")
    parser.add_argument('--output_path', type=str, default="./prediction/submission.csv")

    args = parser.parse_args()

    if args.config_path != "" :
        config = configparser.ConfigParser()
        config.read(args.config_path)
        args.model_name = config['model']['model_name']

        args.model_path     = config['inference']['saved_model_path']
        args.tokenizer_name = config['inference']['tokenizer_name']
        args.output_path    = config['inference']['output_path']

        main(args, config)
    else:
        main(args)