import pickle as pickle
import os
import pandas as pd
import torch
from preprocessing import *

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
    
        return item

    def __len__(self):
        return len(self.labels)

def preprocessing_dataset(dataset):
    out_dataset = preprocess(dataset)
    return out_dataset

def load_data(dataset_dir, train=True):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset) if train else pd_dataset
  
    return dataset

def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
        
    for v in label:
        try:
            num_label.append(dict_label_to_num[v])
        except:
            num_label.append(100)
  
    return num_label


def tokenized_dataset(dataset_dir, tokenizer, tokenizing_type="type_entity_marker_punct", added_special_tokens=None, train=True):
    """ 위치에 따라 파일을 가져와 dataset을 구성합니다."""
    dataset = load_data(dataset_dir, train)
    
    """ dataset에서 가져온 value값으로 label을 가져옵니다. """
    num_label = label_to_num(dataset['label'].values)

    prepro_data = [] # Tokenizing 된 dataset을 담을 list
    if tokenizing_type == "base":
        # BaseLine code
        """ dataset에서 가져온 entity들을 tokenizing 합니다."""
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
          temp = ''
          temp = dict(eval(e01))['word'] + '[SEP]' + dict(eval(e02))['word']
          prepro_data.append(temp)
    elif tokenizing_type == "entity_marker" and added_special_tokens is not None:
        # 1. Add Special Tokens to Each Entity Side - Entity Marker
        for i in range(len(dataset)):
            bse_idx = dict(eval(dataset['subject_entity'][i]))['start_idx']
            ese_idx = dict(eval(dataset['subject_entity'][i]))['end_idx']
            boe_idx = dict(eval(dataset['object_entity'][i]))['start_idx']
            eoe_idx = dict(eval(dataset['object_entity'][i]))['end_idx']
            if bse_idx < boe_idx:
                prepro_data.append(
                    dataset['sentence'][i][:bse_idx] +
                    added_special_tokens[0] + dataset['sentence'][i][bse_idx:ese_idx + 1] + added_special_tokens[1] +
                    dataset['sentence'][i][ese_idx + 1:boe_idx] +
                    added_special_tokens[2] + dataset['sentence'][i][boe_idx:eoe_idx + 1] + added_special_tokens[3] +
                    dataset['sentence'][i][eoe_idx + 1:]
                )

            else:
                prepro_data.append(
                    dataset['sentence'][i][:boe_idx] +
                    added_special_tokens[2] + dataset['sentence'][i][boe_idx:eoe_idx + 1] + added_special_tokens[3] +
                    dataset['sentence'][i][eoe_idx + 1:bse_idx] +
                    added_special_tokens[0] + dataset['sentence'][i][bse_idx:ese_idx + 1] + added_special_tokens[1] +
                    dataset['sentence'][i][ese_idx + 1:]
                )
    elif tokenizing_type == "type_entity_marker_punct":
        # 2. Typed Entity Marker
        for i in range(len(dataset)):
            subj, obj = dict(eval(dataset['subject_entity'][i]))['type'], dict(eval(dataset['object_entity'][i]))['type']
            subj = subj.replace("_", " ").lower()
            obj = obj.replace("_", " ").lower()
            subj_token = f"@*{subj}*"
            obj_token = f"#^{obj}^"

            bse_idx = dict(eval(dataset['subject_entity'][i]))['start_idx']
            ese_idx = dict(eval(dataset['subject_entity'][i]))['end_idx']
            boe_idx = dict(eval(dataset['object_entity'][i]))['start_idx']
            eoe_idx = dict(eval(dataset['object_entity'][i]))['end_idx']
            if bse_idx < boe_idx:
                prepro_data.append(dataset['sentence'][i][:bse_idx] +
                                   subj_token + dataset['sentence'][i][bse_idx:ese_idx + 1] + "@" +
                                   dataset['sentence'][i][ese_idx + 1:boe_idx] +
                                   obj_token + dataset['sentence'][i][boe_idx:eoe_idx + 1] + "#" +
                                   dataset['sentence'][i][eoe_idx + 1:])
            else:
                prepro_data.append(dataset['sentence'][i][:boe_idx] +
                                   obj_token + dataset['sentence'][i][boe_idx:eoe_idx + 1] + "#" +
                                   dataset['sentence'][i][eoe_idx + 1:bse_idx] +
                                   subj_token + dataset['sentence'][i][bse_idx:ese_idx + 1] + "@" +
                                   dataset['sentence'][i][ese_idx + 1:])
    else:
        raise NameError(f"{tokenizing_type} is not defined. Please select one of ['base', 'entity_marker', 'type_entity_marker_punct']")

    tokenized_sentences = tokenizer(
        prepro_data,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    
    return [tokenized_sentences,num_label]

def tokenized_dataset_with_entity(dataset_dir, tokenizer, train=True):
    """ 위치에 따라 파일을 가져와 dataset을 구성합니다. """
    dataset = load_data(dataset_dir, train)
    
    """ dataset에서 가져온 value값으로 label을 가져옵니다. """
    num_label = label_to_num(dataset['label'].values)
  
    """ dataset에서 가져온 entity들을 tokenizing 합니다. """
    concat_entity = []
    entities = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        entities.append([e01, e02])
        concat_entity.append(temp)
        
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
 
    entity_ids = []
    for idx, sentence in enumerate(list(dataset['sentence'])) :
        entity_id = [0 for i in range(241)]
        tokens = tokenizer.tokenize(sentence)
        entities[idx] = tokenizer.tokenize(entities[idx])

        for id, token in enumerate(tokens):
            entity_id[id] = int(token in entities[idx])
            
        entity_ids.append(torch.tensor(entity_id))
    
    tokenized_sentences['entity_ids'] = entity_ids
    
    return [tokenized_sentences,num_label]