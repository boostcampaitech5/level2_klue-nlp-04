import pandas as pd
import configparser

'''
    *현재 load_data.py에서 argument를 받지 않아서 preprocessing.py에서 직접 config 파일을 참조하고 있음
    *따라서 경로가 바뀌면 오류가 날 수 있음
'''

config = configparser.ConfigParser()
config.read('config.conf')


def preprocess(dataframe):
    if config.getboolean('preprocessing','use_clean_duplicate'):
        print('preprocessing - use_clean_duplicate')
        dataframe = clean_duplicate(dataframe)
    if config.getboolean('preprocessing','use_clean_no_relation_word_pair'):
        print('preprocessing - use_clean_no_relation_word_pair')
        dataframe = clean_no_relation_word_pair(dataframe)
    return dataframe


def clean_duplicate(dataframe):
    """
        입력 받는 dataframe의 'sentence', 'subject_entity', 'object_entity', 'label'의 동일한 값이
        여러 존재할 때 한개의 값만 남기고 삭제함
        
        입력 받는 dataframe의 'sentence', 'subject_entity', 'object_entity'의 값이 동일하지만 label 값이
        서로 다른 경우 'no_relation', 'org:members'이 포함된 label 값을 제거함
        Args:
            dataframe: pandas dataframe
        Returns:
            cleaned: 전처리한 dataframe. 
    """
    duplicated = dataframe[dataframe.duplicated(['sentence', 'subject_entity', 'object_entity'], keep=False)]
    groups = duplicated.groupby(['sentence', 'subject_entity', 'object_entity'])
    index_list = []
    for name, group in groups:
        unique_labels = group['label'].unique()
        if len(unique_labels) > 1:
            index_list.extend(group.index.tolist())
    idx = dataframe.iloc[index_list,:][dataframe.iloc[index_list,:]['label'].isin(['no_relation', 'org:members'])].index
    cleaned = dataframe.drop(idx)
    cleaned = cleaned.drop_duplicates(['sentence', 'subject_entity', 'object_entity'], keep='first')
    cleaned = cleaned.reset_index(drop=True)
    return cleaned
    

def clean_no_relation_word_pair(dataframe):
    """
        'subject_entity'의 word 와 'object_entity'의 word의 조합이 동일한 데이터들 중 label이 no_relation 일 때
        처음 등장한 데이터만 남기고 나머지를 삭제함
        Args:
            dataframe: pandas dataframe
        Returns:
            cleaned: 전처리한 dataframe. 
    """
    print('clean_no_relation_word_pair')
    dict_word_pair={}
    for i in range(dataframe.shape[0]):
        if dataframe.iloc[i,:]['label'] == 'no_relation':
            sub_word=eval(dataframe.iloc[i,:]['subject_entity'])['word']
            obj_word=eval(dataframe.iloc[i,:]['object_entity'])['word']
            key = ','.join([sub_word, obj_word])
            nums, labels, idxs = dict_word_pair.get(key, (0,[],[]))
            labels.append(dataframe.iloc[i,:]['label'])
            idxs.append(i)
            dict_word_pair[key] = (nums+1, labels, idxs)
    idx_arr=[]
    for key, value in dict_word_pair.items():
        nums, labels, idxs = value
        if len(labels) >= 2 and len(set(labels)) == 1 and labels[0] == 'no_relation':
            idx_arr.extend(idxs[1:])
    cleaned = dataframe.drop(idx_arr)
    cleaned = cleaned.reset_index(drop=True)
    return cleaned