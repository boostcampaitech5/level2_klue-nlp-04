import sklearn
import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_seperate_f1(preds, labels):
    """KLUE-RE seperate f1 (including no_relation)
    
    출력물은 #precision, #recall, f1-score, support(=개수)로 2개입니다.
    """
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    # no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    metric = sklearn.metrics.precision_recall_fscore_support(
        labels, preds, average=None, labels=label_indices)
    metric_label = ['precision', 'recall', 'f1-score', 'support']
    ans = [ (metric_label[i], list(metric[i])) for i in range(len(metric_label)) ]
    ans[2] = (ans[2][0], [ (label_list[i], v) for i, v in enumerate(ans[2][1]) ] )
    ans[3] = (ans[3][0], [ (label_list[i], v) for i, v in enumerate(ans[3][1]) ] )
    ans.pop(0)
    ans.pop(0)
    return ans


def draw_cofusion_matrix(preds, labels):
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
        'org:product', 'per:title', 'org:alternate_names',
        'per:employee_of', 'org:place_of_headquarters', 'per:product',
        'org:number_of_employees/members', 'per:children',
        'per:place_of_residence', 'per:alternate_names',
        'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
        'per:spouse', 'org:founded', 'org:political/religious_affiliation',
        'org:member_of', 'per:parents', 'org:dissolved',
        'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
        'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
        'per:religion']
    mat = sklearn.metrics.confusion_matrix(
        y_true=labels, y_pred=preds, normalize='true')
    mat = np.array([list(int('{:2.0f}'.format((x*99))) for x in xx) for xx in mat])
    #0.02로 적힌 숫자를 100을 곱해서 2로 만들어주고 내림해줍니다.
    np.set_printoptions(linewidth=150) 
    for i in range(len(mat)):
        print(mat[i], end=' ')
        print(label_list[i])
    np.set_printoptions(linewidth=75)
    return


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  seperate_f1 = klue_re_seperate_f1(preds, labels)
  draw_cofusion_matrix(preds, labels)

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
      'seperate f1 score': seperate_f1
  }