import argparse

from trainer.train import *
from utils.inference.inference import *
import configparser
import json
import wandb

def _decode(o):
    # Note the "unicode" part is only for python2
    if isinstance(o, str):
        try:
            return float(o)
        except ValueError:
            return o
    elif isinstance(o, dict):
        return {k: _decode(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [_decode(v) for v in o]
    else:
        return o

def main(args, config=None):
    if args.mode == "train" :

        if config and config["sweep"].getboolean("run_sweep") :
            print("Training with sweep mode")

            sweep_id = wandb.sweep(json.load(open(config["sweep"]["sweep_path"], "r"), object_hook=_decode), project="RE", entity="nlp-10")
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