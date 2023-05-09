import argparse

from train import *
from inference import *

def main(args):
    if args.mode == "train" :
        print("Training mode")
        main_train()
        
    elif args.mode == "inference" :
        print("Inference mode")
        main_inference("./best_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    args = parser.parse_args()
    main(args)