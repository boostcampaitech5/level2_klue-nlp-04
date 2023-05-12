import pandas as pd
import configparser

'''
    *현재 load_data.py에서 argument를 받지 않아서 preprocessing.py에서 직접 config 파일을 참조하고 있음
    *따라서 경로가 바뀌면 오류가 날 수 있음
'''

config = configparser.ConfigParser()
config.read('config.conf')

use_clean_duplicate = config['preprocessing']['use_clean_duplicate']


def preprocess(dataframe):
    if use_clean_duplicate:
        dataframe = clean_duplicate(dataframe)
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
    