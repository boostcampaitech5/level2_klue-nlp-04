# 🗓️ 개발 기간

2023.05.02 ~ 2023.05.18(총 16일)

# 📄 프로젝트 소개

- 본 대회에서는 문장에 존재하는 두 개체(Entity)의 관계를 예측하는 Relation Extraction Task를 수행함.
- 주어진 문장과 두 개체(Entity)의 관계를 분석하여 30개의 관계 중 하나를 예측함.
- 주어진 데이터셋은 KLUE benchmark의 RE 데이터셋을 기반으로 구성됨.

# 💽 사용 데이터셋

- 데이터셋은 Train Data 32,470개 와 Test Data 7,765개 로 구성됨.
- 각 개체(Entity)는 Subject와 Object로 표기하며, Subject와 Object는 문장 내에서 각각 하나의 개체를 의미함.
- 개체 Columns는 각 개체의 시작 Index와 끝 Index, Word가 주어지며 아래와 같은 형태로 정의됨.
    
    
    | id | 샘플 순서 id |
    | --- | --- |
    | sentence | subject_entity의 단어와 object_entity의 단어가 포함된 문장 |
    | subject_entity | 단어, 단어의 시작 인덱스, 단어의 끝 인덱스 및 유형 정보를 포함한 주체 개체 |
    | object_entity | 단어, 단어의 시작 인덱스, 단어의 끝 인덱스 및 유형 정보를 포함한 목적 개체 |
    | label | 주어진 문장의 주체 개체와 목적 개체 사이의 관계를 나타내는 레이블 |
    | source | 문장의 출처(wikitree, wikipedia, policy_briefing) |
- label은 30개로 분류되어 있고, 아래와 같이 정의됨.
    
    
    | Label | 설명 | Label | 설명 |
    | --- | --- | --- | --- |
    | no_relation | 관계가 존재하지 않는 경우 | per:place_of_death | 인물의 사망지 관계 |
    | org:dissolved | 조직이 해산된 날짜 관계 | per:place_of_residence | 인물의 거주지 관계 |
    | org:founded | 조직이 창립된 날짜 관계 | per:origin | 인물의 출신 관계 |
    | org:place_of_headquarters | 조직의 본사 위치 관계 | per:employee_of | 인물이 다니는 조직 관계 |
    | org:alternate_names | 조직의 다른 이름 관계 | per:schools_attended | 인물이 다닌 학교 관계 |
    | org:member_of | 조직의 구성원 관계 | per:alternate_names | 인물의 다른 이름 관계 |
    | org:members | 조직의 구성원 관계 | per:parents | 인물의 부모 관계 |
    | org:political/religious_affiliation | 조직의 종교나 정치적 성향 관계 | per:children | 인물의 자녀 관계 |
    | org:product | 회사의 제품과 관련된 관계 | per:siblings | 인물의 형제자매 관계 |
    | org:founded_by | 조직을 창립한 인물 관계 | per:spouse | 인물의 배우자 관계 |
    | org:top_members/employees | 조직을 대표하는 인물(임원) 관계 | per:other_family | 인물의 가족 관계 |
    | org:number_of_employees/members | 조직의 구성원 수 관계 | per:colleagues | 인물의 동료 관계 |
    | per:date_of_birth | 인물의 생년월일 관계 | per:product | 인물이 출시한 제품 관계 |
    | per:date_of_death | 인물의 사망일 관계 | per:religion | 인물의 종교 관계 |
    | per:place_of_birth | 인물의 출생지 관계 | per:title | 인물의 이름 관계 |

# 📋 평가 지표

- **Micro F1-Score** : ****micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여. 데이터 분포상 많은 부분을 차지하고 있는 no_relation class는 제외하고 F1 score가 계산됨.
- **AUPRC** : PRC아래의 면적값으로, 모든 class에 대한 평균적인 AUPRC로 계산해 score를 측정함. imbalance한 데이터에 유용함.

# 👨‍👨‍👧‍👧 멤버 구성 및 역할

| [곽민석](https://github.com/kms7530) | [이인균](https://github.com/lig96) | [임하림](https://github.com/halimx2) | [최휘민](https://github.com/ChoiHwimin) | [황윤기](https://github.com/dbsrlskfdk) |
| --- | --- | --- | --- | --- |
| <img src="https://avatars.githubusercontent.com/u/6489395" width="140px" height="140px" title="Minseok Kwak" /> | <img src="https://avatars.githubusercontent.com/u/126560547" width="140px" height="140px" title="Ingyun Lee" /> | <img src="https://ca.slack-edge.com/T03KVA8PQDC-U04RK3E8L3D-ebbce77c3928-512" width="140px" height="140px" title="Halim Lim" /> | <img src="https://avatars.githubusercontent.com/u/102031218?v=4" width="140px" height="140px" title="ChoiHwimin" /> | <img src="https://avatars.githubusercontent.com/u/4418651?v=4" width="140px" height="140px" title="yungi" /> |
- **곽민석**
    - 실험을 위한 Sweep 및 config 추가
    - 하이퍼 파라미터 튜닝
    - 이진분류 및 세부 분류 모델 모델링
    - 모델 탐색 및 실험
    - 코드 개선
- **이인균**
    - EDA
    - 불균형 데이터에 따른 weighted loss 구현
    - Label smoothing 구현
    - Loss Function 구현
- **임하림**
    - 모듈화 작업
    - Bert-base, RoBERTa에 embedding layer 추가하기
    - T5 모델링
    - 하이퍼 파라미터 튜닝
    - 코드 리팩토링
- **최휘민**
    - EDA
    - 전처리 성능 실험(중복 데이터 제거, 한자처리)
    - 일반화 성능 실험(중복 단어 조합 제거, Downsampling)
    - Data Augmentation 실험(Easy Data Augmentation 기법)
    - SOTA 모델 탐색, 모델 실험, 예측 결과 분석
- **황윤기**
    - 논문 리서치
    - Data Split
    - Modeling(RBERT, Improved Baseline)
    - 전처리 구현(Add Contexts, Typed Entity Marker Punct.)
    - 하이퍼 파라미터 튜닝
    - Github Projects 환경 구성
    - Loss Function 리서치 및 구현

# ⚒️ 기능 및 사용 모델

- `RBERT` : KLUE/RoBERTa Large + [CLS] Embedding + [subj] Entitiy Embedding Avg + [obj] Entity Embedding Avg + Classifier
- 문장의 시작 부분에 아래와 같이 자연어로 관계에 대한 서술을 추가함.

```bash
subject Entity 는 {subj} 이다. object Entity 는 {obj} 이다. {subj} 는 사
람이다. {obj}는 장소이다.
```

# 🏗️ 프로젝트 구조

```bash
├── README.md
├── code
│   ├── README.md
│   ├── config
│   │   ├── config.conf
│   │   └── sweep.json
│   ├── data
│   │   ├── dict_label_to_num.pkl
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── img.png
│   ├── main.py
│   ├── model
│   │   ├── model.py
│   │   ├── modeling_MT5.py
│   │   ├── modeling_bert.py
│   │   └── modeling_roberta.py
│   ├── trainer
│   │   ├── train.py
│   │   └── train_entity_embedding.py
│   └── utils
│       ├── CustomScheduler.py
│       ├── inference
│       │   ├── dict_num_to_label.pkl
│       │   └── inference.py
│       ├── metrics.py
│       └── requirements.txt
└── img.png
```

# 🔗 링크

- [Wrap-up report](/assets/docs/NLP_04_Wrap-Up_Report_RE.pdf)
