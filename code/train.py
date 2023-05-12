import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
import time
import random
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, AdamW

from load_data import *
from metrics import *
from torch.utils.data import random_split
from CustomScheduler import CosineAnnealingWarmUpRestarts


def _getTrainerWithConfig(config):
    return TrainingArguments(
        fp16                            = config["train"]["fp16"],
        gradient_checkpointing          = config["train"]["gradient_checkpointing"],
        output_dir                      = config["train"]["output_dir"],
        save_total_limit                = int(config["train"]["save_total_limit"]),
        save_steps                      = int(config["train"]["save_steps"]),
        num_train_epochs                = int(config["train"]["num_train_epochs"]),
        learning_rate                   = float(config["train"]["learning_rate"]),
        per_device_eval_batch_size      = int(config["train"]["per_device_eval_batch_size"]),
        warmup_steps                    = int(config["train"]["warmup_steps"]),
        weight_decay                    = float(config["train"]["weight_decay"]),
        logging_dir                     = config["train"]["logging_dir"],
        logging_steps                   = int(config["train"]["logging_steps"]),
        eval_steps                      = int(config["train"]["eval_steps"]),
        evaluation_strategy             = config["train"]["evaluation_strategy"],
        load_best_model_at_end          = config["train"]["load_best_model_at_end"]
    )
  
  
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    
def train(args, config=None):
    seed_everything(42)

    MODEL_NAME = "klue/roberta-small"

    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Entitiy Marker 사용
    # added_special_tokens = ['[SE]', '[/SE]', '[OE]', '[/OE]']
    # added_token_num = tokenizer.add_special_tokens({'additional_special_tokens': added_special_tokens})


    # Data Load and Tokenizing
    # 1. Entity Marker 사용
    # tokenized_total, total_label = tokenized_dataset("../dataset/train/train.csv", tokenizer, tokenizing_type="entity_marker", added_special_tokens)
    # 2. Base, typed Entity Marker Punct 사용
    tokenized_total, total_label = tokenized_dataset("../dataset/train/train.csv", tokenizer, tokenizing_type="type_entity_marker_punct")


    # split dataset for pytorch.
    train_frac = 0.8
    RE_total_dataset = RE_Dataset(tokenized_total, total_label)
    RE_train_dataset, RE_val_dataset = random_split(RE_total_dataset, [int(len(RE_total_dataset)*train_frac), len(RE_total_dataset)-int(len(RE_total_dataset)*train_frac)])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30
    
    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    # Entity Marker 사용
    # model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 추가한 Special token 갯수만큼 Embedding을 늘려줘야함
    print(model.config)
    model.parameters
    model.to(device)

    # optimizer and scheduler
    optimizers = AdamW(model.parameters(), lr=0)
    scheduler = CosineAnnealingWarmUpRestarts(optimizers, T_0=1000, T_mult=2, eta_max=3e-5,  T_up=500, gamma=0.5)

    training_args = TrainingArguments(
        fp16=True,                      # use 16-bit (mixed) precision to increase speed
        gradient_checkpointing=True,    # use gradient checkpointing to reduce memory usage
        # TODO:
        # adamw 대신 adafactor, 8비트아담
        # 데이터로더 pin과 nun_worker 설정하기
        # 배치 크기 늘리기
        ##
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step. ## check-point가 여기야
        num_train_epochs=5,              # total number of training epochs
        learning_rate=5e-5,               # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                     # `no`: No evaluation during training.
                                     # `steps`: Evaluate every `eval_steps`.
                                     # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.

        load_best_model_at_end = True 
    )

    training_args = _getTrainerWithConfig(config) if config else training_args
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_val_dataset,             # evaluation dataset ## 수정
        compute_metrics=compute_metrics,        # define metrics function
        
        optimizers=(optimizers, scheduler)
        )

    # train model
    trainer.train()
    
    # 모델 저장
    # 모델 저장 경로와 이름 설정
    model_save_path = './best_model'
    import datetime
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(KST)
    now = now.strftime("%mM%dD%HH%MM")
    model_name = 'model_{}_{}'.format(MODEL_NAME, now)

    # 경로와 이름을 합쳐서 완전한 경로 생성
    model_path = os.path.join(model_save_path, model_name)

    # 모델 저장 경로에 폴더가 없으면 폴더 생성
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # 모델 저장
    model.save_pretrained(model_path)
