import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
import wandb
import random
import transformers

from collections import namedtuple
from transformers import AdamW, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from metrics import *
from torch.utils.data import random_split
from CustomScheduler import CosineAnnealingWarmUpRestarts
from model import Model

def seed_everything(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  seed_everything(42)
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  # MODEL_NAME = "tunib/electra-ko-base"
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # # Entitiy Marker 사용
  # added_special_tokens = ['[SE]', '[/SE]', '[OE]', '[/OE]']
  # added_token_num = tokenizer.add_special_tokens({'additional_special_tokens': added_special_tokens})

  # load dataset
  train_dataset = load_data("../dataset/train/train.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)

# tokenizing dataset
  # Entity Marker 사용
  # tokenized_train = tokenized_dataset(train_dataset, tokenizer, added_special_tokens)
  # Typed Entitiy Marker 사용
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

# make dataset for pytorch.
  RE_total_dataset = RE_Dataset(tokenized_train, train_label)
  RE_train_dataset, RE_val_dataset = random_split(RE_total_dataset, [int(len(RE_total_dataset)*0.8), int(len(RE_total_dataset)*0.2)])
  # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)

  # setting model Arguments
  Args = namedtuple("Args", ["SUBJ_TOKEN", "OBJ_TOKEN"])
  args = Args(tokenizer.convert_tokens_to_ids("@"), tokenizer.convert_tokens_to_ids("#"))
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

#   model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
# # # Entity Marker 사용
#   # model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 추가한 Special token 갯수만큼 Embedding을 늘려줘야함
#   print(model.config)
#   model.parameters
  model = Model(model_config, args)
  model.to(device)
  optimizers = AdamW(model.parameters(), lr=0)
  scheduler = CosineAnnealingWarmUpRestarts(optimizers, T_0=1000, T_mult=2, eta_max=3e-5,  T_up=200, gamma=0.5)

  wandb.login()
  run = wandb.init(project="NER_TEST", entity="dbsrlskfdk")
  run.name = MODEL_NAME
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=5,              # total number of training epochs
    # learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    gradient_accumulation_steps=4,   # gradient accumulation steps
    # fp16=True,                        # Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
    # lr_scheduler_type="cosine_with_restarts",
    # warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to="wandb",
    disable_tqdm=True
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_val_dataset,             # evaluation dataset
    optimizers=(optimizers, scheduler),
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
  run.finish()

# if __name__== '__main__':
#   train()
