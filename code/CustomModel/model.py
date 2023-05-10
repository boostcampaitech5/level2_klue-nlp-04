import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, RobertaPreTrainedModel


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


class Model(RobertaPreTrainedModel):
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


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        output = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        sequence_output = output[0] # Sequence Representation
        pooled_output = output[1] # [CLS]

        indices = torch.arange(0, 2 * input_ids.size(0), 2).unsqueeze(axis=-1).to(input_ids.device)
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        subj_start = (input_ids == self.SUBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 SUBJ Token의 index를 찾음(Subject 시작을 의미)
        obj_start = (input_ids == self.OBJ_TOKEN).nonzero()[indices, 1].view(-1) # 처음 발견되는 OBJ Token의 index를 찾음(Object 시작을 의미)

        pooled_output = self.cls_fc_layer(pooled_output) # [CLS]의 hidden state(representation)를 FC Layer에 통과
        subj_h = self.entity_fc_layer(sequence_output[idx, subj_start]) # Subject의 hidden state(representation)를 FC Layer에 통과
        obj_h = self.entity_fc_layer(sequence_output[idx, obj_start]) # Object의 hidden state(representation)를 FC Layer에 통과

        # h = torch.cat((sequence_output[idx, subj_start], sequence_output[idx, obj_start]), dim=-1) # Subject와 Object의 hidden state(representation)를 concat
        h = torch.cat([pooled_output, subj_h, obj_h], dim=-1) # Subject와 Object의 hidden state(representation)를 concat

        logits = self.classifier(h) # classifier의 Input으로 사용

        outputs = (logits,) + output[2:] # logits, (hidden_states), (attentions)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs