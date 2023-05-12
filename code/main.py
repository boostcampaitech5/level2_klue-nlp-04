import argparse

from train import *
from inference import *
import configparser
import json
import wandb

def main(args, config=None):
    if args.mode == "train" :

        if config["sweep"].getboolean("run_sweep") :
            print("Training with sweep mode")

            sweep_id = wandb.sweep(json.load(open(config["sweep"]["sweep_path"], "r")), project="RE", entity="nlp-10")
            wandb.agent(sweep_id, trainWithSweep, count=int(config["sweep"]["sweep_count"]))
            wandb.finish()
        else:
            wandb.init(
                project="RE",
                entity="nlp-10"
            )

            print("Training mode")
            train(args, config)
            
            wandb.finish()
        
    elif args.mode == "inference" :
        print("Inference mode")
        main_inference(args)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--model_dir', type=str, default="./best_model")
    parser.add_argument('--model_name', type=str, default="klue/bert-base")
    parser.add_argument('--config_path', type=str, default="")

    args = parser.parse_args()

    if args.config_path != "" :
        config = configparser.ConfigParser()
        config.read(args.config_path)
        args.model_name = config['train']['model_name']

        main(args, config)
    else:
        main(args)