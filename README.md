# Relation Extraction Task Competition
## Introduction
- 본 대회에서는 문장에 존재하는 두 개체(Entity)의 관계를 예측하는 Relation Extraction Task를 수행합니다.
- 주어진 문장과 두 개체(Entity)의 관계를 분석하여 30개의 관계 중 하나를 예측합니다.
- 주어진 데이터셋은 KLUE benchmark의 RE 데이터셋을 기반으로 합니다.

## Data Description
- 데이터셋은 Train Data 32,470개 와 Test Data 7,765개 로 구성되어 있습니다.
- 각 개체(Entity)는 Subject와 Object로 표기하며, Subject와 Object는 문장 내에서 각각 하나의 개체를 의미합니다.
- 개체 Columns는 각 개체의 시작 Index와 끝 Index, Word가 주어집니다.
- label은 30개로 분류되어 있습니다.
- 각 label은 아래와 같이 정의됩니다.
  - `no_relation`: 관계가 존재하지 않는 경우
  - `org:dissolved`: 조직이 해산된 날짜 관계
  - `org:founded`: 조직이 창립된 날짜 관계
  - `org:place_of_headquarters`: 조직의 본사 위치 관계
  - `org:alternate_names`: 조직의 다른 이름 관계
  - `org:member_of`: 조직의 구성원 관계
  - `org:members` : 조직의 구성원 관계
  - `org:political/religious_affiliation`: 조직의 종교나 정치적 성향 관계
  - `org:product`: 회사의 제품과 관련된 관계
  - `org:founded_by`: 조직을 창립한 인물 관계
  - `org:top_members/employees`: 조직을 대표하는 인물(임원) 관계
  - `org:number_of_employees/members`: 조직의 구성원 수 관계
  - `per:date_of_birth`: 인물의 생년월일 관계
  - `per:date_of_death`: 인물의 사망일 관계
  - `per:place_of_birth`: 인물의 출생지 관계
  - `per:place_of_death`: 인물의 사망지 관계
  - `per:place_of_residence`: 인물의 거주지 관계
  - `per:origin`: 인물의 출신 관계
  - `per:employee_of`: 인물이 다니는 조직 관계
  - `per:schools_attended`: 인물이 다닌 학교 관계
  - `per:alternate_names`: 인물의 다른 이름 관계
  - `per:parents`: 인물의 부모 관계
  - `per:children`: 인물의 자녀 관계
  - `per:siblings`: 인물의 형제자매 관계
  - `per:spouse`: 인물의 배우자 관계
  - `per:other_family`: 인물의 가족 관계
  - `per:colleagues`: 인물의 동료 관계
  - `per:product`: 인물이 출시한 제품 관계
  - `per:religion`: 인물의 종교 관계
  - `per:title`: 인물의 이름 관계

## Data Example
- 아래는 주어진 데이터셋의 예시입니다.
![img.png](img.png)


## Submission History
