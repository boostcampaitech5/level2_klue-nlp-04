import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig, RobertaPreTrainedModel, ElectraPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, activation=True) -> None:
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.activation:
            x = self.tanh(x)
        return self.linear(x)

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss Function
    References:
        https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters:
            x: input logits
            y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class TwoEntityModel(RobertaPreTrainedModel):
    def __init__(self, config, args) -> None:
        super().__init__(config)
        # setting model hyperparameter
        # self.model_config = AutoConfig.from_pretrained(MODEL_NAME)
        self.plm = AutoModel.from_pretrained(config.name_or_path, config=config)

        self.num_labels = config.num_labels
        self.SUBJ_TOKEN = args.SUBJ_TOKEN
        self.OBJ_TOKEN = args.OBJ_TOKEN

        self.loss_fct = nn.CrossEntropyLoss()

        # # Entity Marker 사용
        # model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 추가한 Special token 갯수만큼 Embedding을 늘려줘야함

        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate=0.1)
        self.classifier = FCLayer(
            2 * config.hidden_size,
            config.num_labels,
            dropout_rate=0.1,
            activation=False,
        )

        print(self.config)
        print(self.parameters)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        output = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        sequence_output = output[0] # Sequence Representation

        indices = torch.arange(0, 2 * input_ids.size(0), 2).unsqueeze(axis=-1).to(input_ids.device)
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        subj_start = (input_ids == self.SUBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 SUBJ Token의 index를 찾음(Subject 시작을 의미)
        obj_start = (input_ids == self.OBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 OBJ Token의 index를 찾음(Object 시작을 의미)

        subj_h = self.entity_fc_layer(sequence_output[idx, subj_start]) # Subject의 hidden state(representation)를 FC Layer에 통과
        obj_h = self.entity_fc_layer(sequence_output[idx, obj_start]) # Object의 hidden state(representation)를 FC Layer에 통과

        h = torch.cat([subj_h, obj_h], dim=-1) # Subject와 Object의 hidden state(representation)를 concat

        logits = self.classifier(h) # classifier의 Input으로 사용

        outputs = (logits,) + output[2:] # logits, (hidden_states), (attentions)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs



class RBertModel(RobertaPreTrainedModel):
    def __init__(self, config, args) -> None:
        super().__init__(config)
        # setting model hyperparameter
        # self.model_config = AutoConfig.from_pretrained(MODEL_NAME)

        self.plm = AutoModel.from_pretrained(config.name_or_path, config=config)

        self.num_labels = config.num_labels
        self.SUBJ_TOKEN = args.SUBJ_TOKEN
        self.OBJ_TOKEN = args.OBJ_TOKEN
        self.loss_fct = AsymmetricLoss()

        # # Entity Marker 사용
        # model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 추가한 Special token 갯수만큼 Embedding을 늘려줘야함

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate=0.1)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate=0.1)
        self.classifier = FCLayer(
            3 * config.hidden_size,
            config.num_labels,
            dropout_rate=0.1,
            activation=False,
        )

        print(self.config)
        print(self.parameters)

    def entity_average(self, input_ids, hidden_output, token):

        """
        Token이 포함된 범위의 Average Vector를 구하는 함수

        Args:
            input_ids: Entity Marker Punc.가 포함된 input_ids
            hidden_output: Sequence Representation
            token: Average Vector를 구하고자 하는 Token

        Returns:
            Averaged Vector
        """
        start_end = (input_ids == token).nonzero()[:, 1]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        mask = torch.zeros_like(input_ids).to(input_ids.device)
        for i in idx:
            mask[i].index_fill_(0, torch.arange(start_end[2 * i], start_end[2 * i + 1]).to(input_ids.device), 1)

        mask_unsqueeze = mask.unsqueeze(1)
        length_tensor = (mask != 0).sum(dim=1).unsqueeze(1).to(input_ids.device)
        sum_vector = torch.bmm(mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector.to(input_ids.device)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        output = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        sequence_output = output[0] # Sequence Representation
        pooled_output = output[1] # [CLS]

        # Entity 시작 토큰만 주어지는 방법
        # indices = torch.arange(0, 2 * input_ids.size(0), 2).unsqueeze(axis=-1).to(input_ids.device)
        # idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        # subj_start = (input_ids == self.SUBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 SUBJ Token의 index를 찾음(Subject 시작을 의미)
        # obj_start = (input_ids == self.OBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 OBJ Token의 index를 찾음(Object 시작을 의미)

        pooled_output = self.cls_fc_layer(pooled_output) # [CLS]의 hidden state(representation)를 FC Layer에 통과
        subj_h = self.entity_average(input_ids, sequence_output, self.SUBJ_TOKEN) # Subject Entity 전체의 representation을 average
        obj_h = self.entity_average(input_ids, sequence_output, self.OBJ_TOKEN) # Object Entity 전체의 representation을 average
        subj_h = self.entity_fc_layer(subj_h) # Subject의 hidden state(representation)를 FC Layer에 통과
        obj_h = self.entity_fc_layer(obj_h) # Object의 hidden state(representation)를 FC Layer에 통과

        h = torch.cat([pooled_output, subj_h, obj_h], dim=-1) # Subject와 Object의 hidden state(representation)를 concat

        logits = self.classifier(h) # classifier의 Input으로 사용

        outputs = (logits,) + output[2:] # logits, (hidden_states), (attentions)

        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(logits, labels)
            # For Asymmetric Loss
            labels = F.one_hot(labels, self.num_labels)
            # labels = labels * 0.9 + 0.1/30 ##label smoothing factor 0.1
            # 성능 향상이 애매해서 주석처리 하겠습니다.
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs
      
      
class ElectraModel(ElectraPreTrainedModel):
    def __init__(self, config, args) -> None:
        super().__init__(config)
        # setting model hyperparameter
        self.plm = AutoModel.from_pretrained(config.name_or_path, config=config)

        self.num_labels = config.num_labels
        self.SUBJ_TOKEN = args.SUBJ_TOKEN
        self.OBJ_TOKEN = args.OBJ_TOKEN

        # # Entity Marker 사용
        # model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 추가한 Special token 갯수만큼 Embedding을 늘려줘야함

        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate=0.1)
        self.classifier = FCLayer(
            2 * config.hidden_size,
            config.num_labels,
            dropout_rate=0.1,
            activation=False,
        )

        print(self.config)
        print(self.parameters)

    def entity_average(self, input_ids, hidden_output, token):
        start_end = (input_ids == token).nonzero()[:, 1]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        mask = torch.zeros_like(input_ids).to(input_ids.device)
        for i in idx:
            mask[i].index_fill_(0, torch.arange(start_end[2 * i], start_end[2 * i + 1]).to(input_ids.device), 1)

        mask_unsqueeze = mask.unsqueeze(1)
        length_tensor = (mask != 0).sum(dim=1).unsqueeze(1).to(input_ids.device)
        sum_vector = torch.bmm(mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector.to(input_ids.device)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        output = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        sequence_output = output[0] # Sequence Representation

        # Entity 시작 토큰만 주어지는 방법
        # indices = torch.arange(0, 2 * input_ids.size(0), 2).unsqueeze(axis=-1).to(input_ids.device)
        # idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        # subj_start = (input_ids == self.SUBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 SUBJ Token의 index를 찾음(Subject 시작을 의미)
        # obj_start = (input_ids == self.OBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 OBJ Token의 index를 찾음(Object 시작을 의미)

        subj_h = self.entity_average(input_ids, sequence_output, self.SUBJ_TOKEN) # Subject Entity 전체의 representation을 average
        obj_h = self.entity_average(input_ids, sequence_output, self.OBJ_TOKEN) # Object Entity 전체의 representation을 average
        subj_h = self.entity_fc_layer(subj_h) # Subject의 hidden state(representation)를 FC Layer에 통과
        obj_h = self.entity_fc_layer(obj_h) # Object의 hidden state(representation)를 FC Layer에 통과

        # h = torch.cat((sequence_output[idx, subj_start], sequence_output[idx, obj_start]), dim=-1) # Subject와 Object의 hidden state(representation)를 concat
        h = torch.cat([subj_h, obj_h], dim=-1) # Subject와 Object의 hidden state(representation)를 concat
        # h = torch.cat([subj_h, obj_h], dim=-1)

        logits = self.classifier(h) # classifier의 Input으로 사용

        outputs = (logits,) + output[2:] # logits, (hidden_states), (attentions)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs