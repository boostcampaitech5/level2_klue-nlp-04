import argparse

from train import *
from inference import *

def main(args):
    if args.mode == "train" :
        print("Training mode")
        train(args)
        
    elif args.mode == "inference" :
        print("Inference mode")
        main_inference(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--model_dir', type=str, default="./best_model")
    parser.add_argument('--model_name', type=str, default="klue/bert-base")

    args = parser.parse_args()
    main(args)