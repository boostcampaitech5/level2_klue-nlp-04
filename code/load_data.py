import pickle as pickle
import os
import pandas as pd
import torch

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
    # """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    # subject_entity = []
    # object_entity = []
    # for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    #     i = i[1:-1].split(',')[0].split(':')[1]
    #     j = j[1:-1].split(',')[0].split(':')[1]
    #
    #     subject_entity.append(i)
    #     object_entity.append(j)
    #
    # out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    out_dataset = dataset
    return out_dataset

def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)
  
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

def tokenized_dataset_inference(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
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
    
    return tokenized_sentences

def tokenized_dataset(dataset_dir, tokenizer):
    """ 위치에 따라 파일을 가져와 dataset을 구성합니다."""
    dataset = load_data(dataset_dir)
    
    """ dataset에서 가져온 value값으로 label을 가져옵니다. """
    num_label = label_to_num(dataset['label'].values)

    # # BaseLine code
    # """ dataset에서 가져온 entity들을 tokenizing 합니다."""
    # concat_entity = []
    # for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    #     temp = ''
    #     temp = e01 + '[SEP]' + e02
    #     concat_entity.append(temp)
    # --------------------------------------------
    # # 1. Add Special Tokens to Each Entity Side - Entity Marker
    # prepro_data = []
    # for i in range(len(dataset)):
    #   bse_idx = dict(eval(dataset['subject_entity'][i]))['start_idx']
    #   ese_idx = dict(eval(dataset['subject_entity'][i]))['end_idx']
    #   boe_idx = dict(eval(dataset['object_entity'][i]))['start_idx']
    #   eoe_idx = dict(eval(dataset['object_entity'][i]))['end_idx']
    #   if bse_idx < boe_idx:
    #     prepro_data.append(
    #       dataset['sentence'][i][:bse_idx] + added_special_tokens[0] + dataset['sentence'][i][bse_idx:ese_idx + 1] +
    #       added_special_tokens[1] + dataset['sentence'][i][ese_idx + 1:boe_idx] + added_special_tokens[2] +
    #       dataset['sentence'][i][boe_idx:eoe_idx + 1] + added_special_tokens[3] + dataset['sentence'][i][eoe_idx + 1:])
    #   else:
    #     prepro_data.append(
    #       dataset['sentence'][i][:boe_idx] + added_special_tokens[2] + dataset['sentence'][i][boe_idx:eoe_idx + 1] +
    #       added_special_tokens[3] + dataset['sentence'][i][eoe_idx + 1:bse_idx] + added_special_tokens[0] +
    #       dataset['sentence'][i][bse_idx:ese_idx + 1] + added_special_tokens[1] + dataset['sentence'][i][ese_idx + 1:])
    # --------------------------------------------
    # 2. Typed Entity Marker
    prepro_data = []
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
