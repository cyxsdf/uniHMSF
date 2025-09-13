import torch
from torch import nn
import torch.nn.functional as F

from modules.unimf import MultimodalTransformerEncoder, MultiScaleCNN
from modules.transformer import TransformerEncoder
from transformers import BertTokenizer, BertModel


class TRANSLATEModel(nn.Module):
    def __init__(self, hyp_params, missing=None):
        """
        Construct a Translate model.
        """
        super(TRANSLATEModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.l_len, self.a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
            self.v_len, self.orig_d_v = 0, 0
        else:
            self.l_len, self.a_len, self.v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.trans_layers = hyp_params.trans_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.trans_dropout = hyp_params.trans_dropout
        self.modalities = hyp_params.modalities  # the input modality
        self.missing = missing  # mark the missing modality

        self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        self.multi = nn.Parameter(torch.Tensor(1, self.embed_dim))
        nn.init.xavier_uniform_(self.multi)

        # translate module
        self.translator = TransformerEncoder(embed_dim=self.embed_dim,
                                             num_heads=self.num_heads,
                                             lens=(self.l_len, self.a_len, self.v_len),
                                             layers=self.trans_layers,
                                             modalities=self.modalities,
                                             missing=self.missing,
                                             attn_dropout=self.attn_dropout,
                                             relu_dropout=self.relu_dropout,
                                             res_dropout=self.res_dropout)

        # project module  # just use fc to replace conv1d :-)
        if 'L' in self.modalities or self.missing == 'L':
            self.proj_l = nn.Linear(self.orig_d_l, self.embed_dim)
        if 'A' in self.modalities or self.missing == 'A':
            self.proj_a = nn.Linear(self.orig_d_a, self.embed_dim)
        if 'V' in self.modalities or self.missing == 'V':
            self.proj_v = nn.Linear(self.orig_d_v, self.embed_dim)

        if self.missing == 'L':
            self.out = nn.Linear(self.embed_dim, self.orig_d_l)
        elif self.missing == 'A':
            self.out = nn.Linear(self.embed_dim, self.orig_d_a)
        elif self.missing == 'V':
            self.out = nn.Linear(self.embed_dim, self.orig_d_v)
        else:
            raise ValueError('Unknown missing modality type')

    def forward(self, src, tgt, phase='train', eval_start=False):
        """
        src and tgt should have dimension [batch_size, seq_len, n_features]
        """
        if self.modalities == 'L':
            if self.missing == 'A':
                x_l, x_a = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)  # (seq, batch, embed_dim)
                x_a = x_a.transpose(0, 1)
            elif self.missing == 'V':
                x_l, x_v = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x_a, x_l = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'V':
                x_a, x_v = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x_v, x_l = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'A':
                x_v, x_a = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_a = x_a.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            (x_l, x_a), x_v = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
        elif self.modalities == 'LV':
            (x_l, x_v), x_a = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
        elif self.modalities == 'AV':
            (x_a, x_v), x_l = src, tgt
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_l = x_l.transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')
        #################################################################################
        # For modal type embedding
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2

        # Prepare the [Uni] token or [Bi] token
        # NOTE: [Uni] or [Bi] is in front of the missing modality
        batch_size = tgt.shape[0]
        multi = self.multi.unsqueeze(1).repeat(1, batch_size, 1)

        if phase != 'test':
            if self.missing == 'L':
                x_l = torch.cat((multi, x_l[:-1]), dim=0)
            elif self.missing == 'A':
                x_a = torch.cat((multi, x_a[:-1]), dim=0)
            elif self.missing == 'V':  # self.missing == 'V'
                x_v = torch.cat((multi, x_v[:-1]), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        else:
            if eval_start:
                if self.missing == 'L':
                    x_l = multi  # use [Uni] or [Bi] token as start to generate missing modality
                elif self.missing == 'A':
                    x_a = multi
                elif self.missing == 'V':
                    x_v = multi
                else:
                    raise ValueError('Unknown missing modality type')
            else:
                if self.missing == 'L':
                    x_l = torch.cat((multi, x_l), dim=0)
                elif self.missing == 'A':
                    x_a = torch.cat((multi, x_a), dim=0)
                elif self.missing == 'V':
                    x_v = torch.cat((multi, x_v), dim=0)
                else:
                    raise ValueError('Unknown missing modality type')

        # Prepare the positional embeddings & modal-type embeddings
        if 'L' in self.modalities or self.missing == 'L':
            x_l_pos_ids = torch.arange(x_l.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            l_pos_embeds = self.position_embeddings(x_l_pos_ids)
            l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_l_pos_ids, L_MODAL_TYPE_IDX))
            l_embeds = l_pos_embeds + l_modal_type_embeds
            x_l = x_l + l_embeds
            x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        if 'A' in self.modalities or self.missing == 'A':
            x_a_pos_ids = torch.arange(x_a.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            a_pos_embeds = self.position_embeddings(x_a_pos_ids)
            a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_a_pos_ids, A_MODAL_TYPE_IDX))
            a_embeds = a_pos_embeds + a_modal_type_embeds
            x_a = x_a + a_embeds
            x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        if 'V' in self.modalities or self.missing == 'V':
            x_v_pos_ids = torch.arange(x_v.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            v_pos_embeds = self.position_embeddings(x_v_pos_ids)
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_v_pos_ids, V_MODAL_TYPE_IDX))
            v_embeds = v_pos_embeds + v_modal_type_embeds
            x_v = x_v + v_embeds
            x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)
        #################################################################################
        # Translation
        if self.modalities == 'L':
            if self.missing == 'A':
                x = torch.cat((x_l, x_a), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_l, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x = torch.cat((x_a, x_l), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_a, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x = torch.cat((x_v, x_l), dim=0)
            elif self.missing == 'A':
                x = torch.cat((x_v, x_a), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            x = torch.cat((x_l, x_a, x_v), dim=0)
        elif self.modalities == 'LV':
            x = torch.cat((x_l, x_v, x_a), dim=0)
        elif self.modalities == 'AV':
            x = torch.cat((x_a, x_v, x_l), dim=0)
        else:
            raise ValueError('Unknown modalities type')

        output = self.translator(x)

        if self.modalities == 'L':
            output = output[self.l_len:].transpose(0, 1)  # (batch, seq, embed_dim)
        elif self.modalities == 'A':
            output = output[self.a_len:].transpose(0, 1)
        elif self.modalities == 'V':
            output = output[self.v_len:].transpose(0, 1)
        elif self.modalities == 'LA':
            output = output[self.l_len + self.a_len:].transpose(0, 1)
        elif self.modalities == 'LV':
            output = output[self.l_len + self.v_len:].transpose(0, 1)
        elif self.modalities == 'AV':
            output = output[self.a_len + self.v_len:].transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')

        output = self.out(output)
        return output


class UNIMFModel(nn.Module):
    def __init__(self, hyp_params):
        """修改后的UniMF模型，使用层次化多尺度时序建模"""
        super(UNIMFModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.orig_l_len, self.orig_a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
        else:
            self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v

        # 多尺度CNN参数
        self.scale_kernel_sizes = hyp_params.scale_kernel_sizes if hasattr(hyp_params, 'scale_kernel_sizes') else [3, 5,
                                                                                                                   7]
        self.scale_count = len(self.scale_kernel_sizes)

        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.multimodal_layers = hyp_params.multimodal_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.modalities = hyp_params.modalities
        self.dataset = hyp_params.dataset
        self.language = hyp_params.language
        self.use_bert = hyp_params.use_bert

        self.distribute = hyp_params.distribute

        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.cls_len = 33
        else:
            self.cls_len = 1
        self.cls = nn.Parameter(torch.Tensor(self.cls_len, self.embed_dim))
        nn.init.xavier_uniform_(self.cls)

        output_dim = hyp_params.output_dim

        # BERT模型
        if self.use_bert:
            self.text_model = BertTextEncoder(language=hyp_params.language, use_finetune=True)

        # 替换原有的Conv1d为多尺度CNN
        self.proj_l = MultiScaleCNN(self.orig_d_l, self.embed_dim, self.scale_kernel_sizes)
        self.proj_a = MultiScaleCNN(self.orig_d_a, self.embed_dim, self.scale_kernel_sizes)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.proj_v = MultiScaleCNN(self.orig_d_v, self.embed_dim, self.scale_kernel_sizes)
        if 'meld' in self.dataset:
            self.proj_cls = nn.Conv1d(self.orig_d_l + self.orig_d_a, self.embed_dim, kernel_size=1)

        # 位置嵌入和模态类型嵌入
        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.position_embeddings = nn.Embedding(max(self.cls_len, self.orig_l_len, self.orig_a_len), self.embed_dim)
        else:
            self.position_embeddings = nn.Embedding(max(self.orig_l_len, self.orig_a_len, self.orig_v_len),
                                                    self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        # 修改后的UniMF，使用层次化多尺度Transformer
        self.unimf = MultimodalTransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            layers=self.multimodal_layers,
            lens=(self.cls_len, self.orig_l_len, self.orig_a_len),
            modalities=self.modalities,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            scale_count=self.scale_count  # 传递尺度数量
        )

        # 投影层
        combined_dim = self.embed_dim
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def forward(self, x_l, x_a, x_v=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # 模态类型嵌入索引
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2
        MULTI_MODAL_TYPE_IDX = 3

        # 准备[CLS] token
        batch_size = x_l.shape[0]
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            cls = self.cls.unsqueeze(1).repeat(1, batch_size, 1)
        else:
            cls = self.proj_cls(torch.cat((x_l, x_a), dim=-1).transpose(1, 2)).permute(2, 0, 1)

        # 位置嵌入和模态类型嵌入
        cls_pos_ids = torch.arange(self.cls_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_l_pos_ids = torch.arange(x_l.shape[1], device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_a_pos_ids = torch.arange(x_a.shape[1], device=x_a.device).unsqueeze(1).expand(-1, batch_size)
        if x_v is not None:
            h_v_pos_ids = torch.arange(x_v.shape[1], device=x_v.device).unsqueeze(1).expand(-1, batch_size)

        cls_pos_embeds = self.position_embeddings(cls_pos_ids)
        h_l_pos_embeds = self.position_embeddings(h_l_pos_ids)
        h_a_pos_embeds = self.position_embeddings(h_a_pos_ids)
        if x_v is not None:
            h_v_pos_embeds = self.position_embeddings(h_v_pos_ids)

        cls_modal_type_embeds = self.modal_type_embeddings(torch.full_like(cls_pos_ids, MULTI_MODAL_TYPE_IDX))
        l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_l_pos_ids, L_MODAL_TYPE_IDX))
        a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_a_pos_ids, A_MODAL_TYPE_IDX))
        if x_v is not None:
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_v_pos_ids, V_MODAL_TYPE_IDX))

        # BERT文本处理
        if self.use_bert:
            x_l = self.text_model(x_l)

        # 多尺度特征提取（替换原有的GRU）
        x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)

        # 多尺度CNN处理
        h_l = self.proj_l(x_l)  # (batch, seq_len, embed_dim)
        h_a = self.proj_a(x_a)
        if x_v is not None:
            x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)
            h_v = self.proj_v(x_v)

        # 转换为(seq_len, batch, embed_dim)格式
        h_l = h_l.transpose(0, 1)
        h_a = h_a.transpose(0, 1)
        if x_v is not None:
            h_v = h_v.transpose(0, 1)

        # 添加位置和模态类型嵌入
        cls_embeds = cls_pos_embeds + cls_modal_type_embeds
        l_embeds = h_l_pos_embeds + l_modal_type_embeds
        a_embeds = h_a_pos_embeds + a_modal_type_embeds
        if x_v is not None:
            v_embeds = h_v_pos_embeds + v_modal_type_embeds

        cls = cls + cls_embeds
        h_l = h_l + l_embeds
        h_a = h_a + a_embeds
        if x_v is not None:
            h_v = h_v + v_embeds

        # 构建多尺度输入
        if x_v is not None:
            # 为每个尺度创建特征序列
            scale_features = []
            for i in range(self.scale_count):
                # 这里简单复制不同模态特征作为多尺度输入
                # 实际应用中可能需要更复杂的尺度划分策略
                scale_seq = torch.cat((cls, h_l, h_a, h_v), dim=0)
                scale_features.append(scale_seq)
        else:
            scale_features = []
            for i in range(self.scale_count):
                scale_seq = torch.cat((cls, h_l, h_a), dim=0)
                scale_features.append(scale_seq)

        # 多模态融合（层次化多尺度Transformer）
        x = self.unimf(scale_features)

        # 输出处理
        if x_v is not None:
            last_hs = x[0]  # 获取[CLS] token
        else:
            last_hs = x[:self.cls_len]  # 获取[CLS] tokens

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        if x_v is None:
            output = output.transpose(0, 1)
        return output, last_hs

class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_bert/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_cn')
            self.model = model_class.from_pretrained('pretrained_bert/bert_cn')

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states