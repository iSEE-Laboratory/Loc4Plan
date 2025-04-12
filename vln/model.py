import torch
from torch import nn
import numpy as np

class ORAR(nn.Module):
    def __init__(self, opts, instr_encoder, image_features=None):
        super(ORAR, self).__init__()
        self.opts = opts
        self.instr_encoder = instr_encoder

        self.dropout = opts.config.dropout
        self.rnn_state_dropout = opts.config.rnn_state_dropout
        self.attn_dropout = opts.config.attn_dropout

        num_heads = opts.config.num_heads
        rnn1_input_size = 16

        if opts.config.use_image_features:
            img_feature_shape = image_features.get('feature_shape', None)

            if self.opts.config.img_feature_dropout > 0:
                self.img_feature_dropout = nn.Dropout(p=self.opts.config.img_feature_dropout)

            img_feature_size = img_feature_shape[-1]
            assert len(img_feature_shape) == 2
            img_feature_flatten_size = img_feature_shape[0] * img_feature_shape[1]

            if img_feature_flatten_size > 2000:
                img_lstm_input_size = 256
                self.linear_img = nn.Linear(img_feature_flatten_size, 512)
                self.img_dropout = nn.Dropout(p=self.dropout)
                self.linear_img_extra = nn.Linear(512, img_lstm_input_size)
                self.img_dropout_extra = nn.Dropout(p=self.dropout)
            else:
                img_lstm_input_size = 64
                self.linear_img = nn.Linear(img_feature_flatten_size, img_lstm_input_size)
                self.img_dropout = nn.Dropout(p=self.dropout)

            rnn1_input_size += img_lstm_input_size

        self.action_embed = nn.Embedding(4, 16)
        if self.opts.config.junction_type_embedding:
            print('use pano embedding')
            self.junction_type_embed = nn.Embedding(4, 16)

        if self.opts.config.junction_type_embedding:
            rnn1_input_size += 16

        if self.opts.config.heading_change:
            rnn1_input_size += 2

        if self.opts.config.rnn1_step:
            rnn1_input_size += 1


        self.rnn = nn.LSTM(input_size=rnn1_input_size,
                           hidden_size=256,
                           num_layers=1,
                           batch_first=True)
        self.rnn_state_h_dropout = nn.Dropout(p=self.dropout)
        self.rnn_state_c_dropout = nn.Dropout(p=self.rnn_state_dropout)

        rnn2_input_size = 256
        if opts.config.use_progress_score:
            rnn2_input_size += 256

        if self.opts.config.use_subInstruction:
            # subInstr
            self.subInstr_attention_layer = nn.MultiheadAttention(embed_dim=256,
                                                                  num_heads=num_heads,
                                                                  dropout=self.attn_dropout,
                                                                  kdim=self.instr_encoder.hidden_size * 2,
                                                                  vdim=self.instr_encoder.hidden_size * 2)
            if self.opts.config.use_layer_norm:
                self.layer_norm_subInstr_attention = nn.LayerNorm(256, eps=1e-6)
            self.subInstr_attn_projection_layer = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU()
            )
            self.subInstr_attn_dropout = nn.Dropout(p=self.dropout)
            self.subInstr_feat_projection_layer = nn.Sequential(
                nn.Linear(self.instr_encoder.hidden_size * 2, 256),
                nn.ReLU()
            )
            self.subInstr_feat_dropout = nn.Dropout(p=self.dropout)
            self.subInstr_projection = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU()
            )
            self.subInstr_dropout = nn.Dropout(p=self.dropout)
            self.node_instr_linear = nn.Linear(256, 1)
            self.node_instr_dropout = nn.Dropout(p=self.dropout)
            self.subInstr_weight = nn.Parameter(torch.randn(1))

        if self.opts.config.use_text_attention:
            self.text_attention_layer = nn.MultiheadAttention(embed_dim=256,
                                                              num_heads=num_heads,
                                                              dropout=self.attn_dropout,
                                                              kdim=self.instr_encoder.hidden_size * 2,
                                                              vdim=self.instr_encoder.hidden_size * 2)
            if self.opts.config.use_layer_norm:
                self.layer_norm_text_attention = nn.LayerNorm(256, eps=1e-6)
            rnn2_input_size += 256

        if opts.config.use_image_features and self.opts.config.use_image_attention:
            self.visual_attention_layer = nn.MultiheadAttention(embed_dim=256,
                                                                num_heads=num_heads,
                                                                dropout=self.attn_dropout,
                                                                kdim=img_feature_size,
                                                                vdim=img_feature_size)
            if self.opts.config.use_layer_norm:
                self.layer_norm_visual_attention = nn.LayerNorm(256, eps=1e-6)
            rnn2_input_size += 256

        self.time_embed = nn.Embedding(self.opts.max_route_len, 32)
        rnn2_input_size += 32

        if self.opts.config.second_rnn:
            print('use second rnn')
            self.rnn2 = nn.LSTM(input_size=rnn2_input_size,
                                hidden_size=256,
                                num_layers=1,
                                batch_first=True)
            self.rnn2_state_h_dropout = nn.Dropout(p=self.dropout)
            self.rnn2_state_c_dropout = nn.Dropout(p=self.rnn_state_dropout)
        else:
            print('use no second rnn')
            self.policy_extra = nn.Linear(rnn2_input_size, 256)
        # self.policy = nn.Linear(256, 4)
        self.policy_forward = nn.Linear(257, 1)
        self.policy_left = nn.Linear(257, 1)
        self.policy_right = nn.Linear(257, 1)
        self.policy_stop = nn.Linear(257, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 进度分数预测
        if self.opts.config.use_progress_score:
            self.progress_projection_layer = nn.Sequential(
                nn.Linear(256,256),
                nn.ReLU()
            )
            self.progress_layer = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            self.progress_dropout = nn.Dropout(p=self.dropout)
        
    def forward(self, text_enc_outputs, text_enc_lengths, token_instr_pair, token_instr_id, max_instr_len, instr_lenghs, subInstr_feat, image_features, a, junction_types, heading_changes, block_progress, intersection_history, h_t, c_t, t, h2_t=None, c2_t=None, step=None, instr_mask=None):
        """
        :param x: [batch_size, 1, 256], encoded instruction
        :param I: [batch_size, 1, 100, 100], features
        :param a: [batch_size, 1], action
        :param p: [batch_size, 1], pano type (street segment, T-intersection, 4-way intersection, >4 neighbors)
        :param h_t: [1, batch_size, 256], hidden state in LSTM
        :param c_t: [1, batch_size, 256], memory in LSTM
        :param t:
        :return:
        """
        # observation layer
        rnn_input = []

        heading_change_1, heading_change_3, action_heading_change = heading_changes

        if self.opts.config.use_image_features:
            if self.opts.config.img_feature_dropout > 0:
                image_features = self.img_feature_dropout(image_features)

            rnn_image_features = image_features.flatten(start_dim=1)
            rnn_image_features = self.linear_img(rnn_image_features)
            rnn_image_features = torch.sigmoid(rnn_image_features)
            rnn_image_features = self.img_dropout(rnn_image_features)
            if hasattr(self, 'linear_img_extra'):
                rnn_image_features = self.linear_img_extra(rnn_image_features)
                rnn_image_features = torch.sigmoid(rnn_image_features)
                rnn_image_features = self.img_dropout_extra(rnn_image_features)
            rnn_input.append(rnn_image_features.unsqueeze(1))

        action_embedding = self.action_embed(a)  # [batch_size, 1, 16]
        rnn_input.append(action_embedding)
        if self.opts.config.junction_type_embedding:
            junction_type_embedding = self.junction_type_embed(junction_types).unsqueeze(1)  # [batch_size, 1, 16]
            rnn_input.append(junction_type_embedding)
        if self.opts.config.heading_change:
            rnn_input.append(heading_change_1)  # [batch_size, 1, 1]
            rnn_input.append(heading_change_3)  # [batch_size, 1, 1]
        if self.opts.config.rnn1_step:
            step = torch.Tensor([step]).repeat(heading_change_3.shape).to(self.device)
            rnn_input.append(step)  # [batch_size, 1, 1]

        s_t = torch.cat(rnn_input, dim=2)
        if h_t is None and c_t is None:  # first timestep
            _, (h_t, c_t) = self.rnn(s_t)
        else:
            _, (h_t, c_t) = self.rnn(s_t, (h_t, c_t))
        h_t = self.rnn_state_h_dropout(h_t)
        c_t = self.rnn_state_c_dropout(c_t)
        trajectory_hidden_state = h_t

        # attention layer
        rnn2_input = [trajectory_hidden_state.squeeze(0)]  # [batch_size, 256]

        progress_score = None
        if self.opts.config.use_progress_score:
            progress_feat = self.progress_projection_layer(h_t)
            progress_score =self.progress_layer(progress_feat)
            progress_score = (progress_score - 0.5) * 2
            progress_feat = self.progress_dropout(progress_feat)
            rnn2_input.append(progress_feat.squeeze(0))


        trajectory_hidden_state_attn_2 = trajectory_hidden_state

        node_instr_score = None
        if self.opts.config.use_subInstruction:
            attn_subInstr_feat, subInstr_attn, subInstr_attn_weights = self.subInstr_attention(
                trajectory_hidden_state, instr_lenghs, subInstr_feat)
            # attn_subInstr_feat, subInstr_attn, subInstr_attn_weights = self.subInstr_attention(
            #     trajectory_hidden_state_attn_2, instr_lenghs, subInstr_feat)

            node_instr_score = self.node_instr_linear(attn_subInstr_feat)
            node_instr_score = self.node_instr_dropout(node_instr_score)
            node_instr_score = nn.Sigmoid()(node_instr_score)
            token_mask_all = torch.Tensor([]).to(self.device)
            for b in range(len(token_instr_id)):
                token_mask = []
                # token_mask_tensor = torch.Tensor([]).to(self.device)
                for instr_idx in range(max_instr_len):
                    if instr_idx >= len(token_instr_id[b]):
                        break
                    else:
                        # expand
                        # token_mask_tensor = torch.concat([token_mask_tensor, node_instr_score[b, instr_idx].expand(token_instr_id[b][instr_idx])], dim=0)
                        token_mask.append(node_instr_score[b, instr_idx].expand(token_instr_id[b][instr_idx]))
                # padding
                padding_len = text_enc_lengths.max() - text_enc_lengths[b]
                token_mask.append(torch.zeros(padding_len).to(self.device))
                # token_mask_tensor = torch.concat([token_mask_tensor, torch.zeros(padding_len).to(self.device)], dim=0)

                token_mask_tensor = torch.concat(token_mask)
                assert token_mask_tensor.shape[0] == text_enc_outputs.shape[0]
                token_mask_all = torch.concat([token_mask_all, token_mask_tensor.unsqueeze(0)], dim=0)
            token_mask_all = token_mask_all.unsqueeze(-1).transpose(0, 1)
            token_mask_all = torch.repeat_interleave(token_mask_all, text_enc_outputs.shape[-1], dim=-1)
            text_enc_outputs_masked = text_enc_outputs * token_mask_all
            text_enc_outputs = text_enc_outputs_masked * self.subInstr_weight + text_enc_outputs

        if self.opts.config.use_text_attention:
            text_attn, text_attn_weights = self._text_attention(trajectory_hidden_state_attn_2, text_enc_outputs, text_enc_lengths)  # [1, batch_size, 256]
            # text_attn, text_attn_weights = self._text_attention(trajectory_hidden_state, text_enc_outputs, text_enc_lengths, instr_mask)  # [1, batch_size, 256]
            rnn2_input.append(text_attn.squeeze(0))  # [batch_size, 256]

        if self.opts.config.use_image_features and self.opts.config.use_image_attention:
            if self.opts.config.use_text_attention:
                image_attn = self._visual_attention(text_attn, image_features)  # [1, batch_size, 256]
            else:
                image_attn = self._visual_attention(trajectory_hidden_state, image_features)  # [1, batch_size, 256]
            rnn2_input.append(image_attn.squeeze(0))  # [batch_size, 256]

        t = self.time_embed(t)
        batch_size = text_enc_lengths.size(0)
        t_expand = torch.zeros(batch_size, 32).to(self.device)
        t_expand.copy_(t)
        rnn2_input.append(t_expand)  # [batch_size, 32]

        rnn2_input = torch.cat(rnn2_input, dim=1)  # [batch_size, 256 + 256 + 256 + 32]
        action_t, (h2_t, c2_t) = self._forward_policy(rnn2_input, h2_t, c2_t, action_heading_change)
        return action_t, (h_t, c_t), (h2_t, c2_t), [text_attn_weights], progress_score, node_instr_score

    def _forward_policy(self, policy_input, h2_t, c2_t, action_heading_change):
        if self.opts.config.second_rnn:
            if h2_t is not None and c2_t is not None:
                _, (h2_t, c2_t) = self.rnn2(policy_input.unsqueeze(1), (h2_t, c2_t))
            else:  # first timestep
                _, (h2_t, c2_t) = self.rnn2(policy_input.unsqueeze(1))
            h2_t = self.rnn2_state_h_dropout(h2_t)
            c2_t = self.rnn2_state_c_dropout(c2_t)
            output_rnn2 = h2_t.squeeze(0)
        else:
            output_rnn2 = self.policy_extra(policy_input)

        final_feat = torch.repeat_interleave(output_rnn2.unsqueeze(1), 4, dim=1)
        final_feat = torch.concat([final_feat, action_heading_change.unsqueeze(-1)], dim=-1)
        # action_t = self.policy(output_rnn2)
        action_t_forward = self.policy_forward(final_feat[:, 0, :]).unsqueeze(1)
        action_t_left = self.policy_left(final_feat[:, 1, :]).unsqueeze(1)
        action_t_right = self.policy_right(final_feat[:, 2, :]).unsqueeze(1)
        action_t_stop = self.policy_stop(final_feat[:, 3, :]).unsqueeze(1)
        action_t = torch.concat([action_t_forward, action_t_left, action_t_right, action_t_stop], dim=1)
        return action_t.squeeze(dim=-1), (h2_t, c2_t)


    def get_node_instr_id(self, node_instr_score, instr_lenghs):
        # node_instr_id
        node_instr_bool = node_instr_score.squeeze(-1) > 0.5
        node_instr_id_all = []
        for b in range(node_instr_score.shape[0]):
            # node_instr_id = [instr_idx if node_instr_bool[b,instr_idx] else for instr_idx in range(instr_lenghs[b])]
            node_instr_id = []
            for instr_idx in range(instr_lenghs[b]):
                if node_instr_bool[b, instr_idx]:
                    node_instr_id.append(instr_idx)
            if len(node_instr_id) == 0:
                node_instr_id = list(range(instr_lenghs[b]))
            node_instr_id_all.append(node_instr_id)
        return node_instr_id_all

    def subInstr_attention(self, trajectory_hidden_state, instr_lenghs, subInstr_feat):
        subInstr_feat = subInstr_feat.transpose(0,1)
        instr_lenghs = torch.tensor(instr_lenghs)
        key_padding_mask = ~(torch.arange(subInstr_feat.shape[0])[None, :] < instr_lenghs[:, None])
        key_padding_mask = key_padding_mask.to(self.device)

        attn, attn_weights = self.subInstr_attention_layer(query=trajectory_hidden_state,
                                                       key=subInstr_feat,
                                                       value=subInstr_feat,
                                                       key_padding_mask=key_padding_mask)
        if self.opts.config.use_layer_norm:
            attn = self.layer_norm_subInstr_attention(attn)

        # attn_subInstr_feat
        attn_proj = self.subInstr_attn_projection_layer(attn)
        attn_proj = self.subInstr_attn_dropout(attn_proj)
        subInstr_feat_proj = self.subInstr_feat_projection_layer(subInstr_feat)
        subInstr_feat_proj = self.subInstr_feat_dropout(subInstr_feat_proj)
        attn_subInstr_feat = attn_proj * subInstr_feat_proj
        attn_subInstr_feat = self.subInstr_projection(attn_subInstr_feat)
        attn_subInstr_feat = self.subInstr_dropout(attn_subInstr_feat)
        # print()
        return attn_subInstr_feat.transpose(0,1), attn, attn_weights


    def _intersection_attention(self, hidden_state, intersection_history, junction_types):
        if len(intersection_history[0]) == 0:
            # step=0
            attn_output = torch.concat([hidden_state, hidden_state], dim=-1)
            return attn_output
        else:
            now_junction_embedding = self.junction_type_embed(junction_types).unsqueeze(0)
            now_action_embedding = self.intersection_attn_cls_token.expand(-1, now_junction_embedding.shape[1], -1)
            now_feat = torch.concat([now_action_embedding, now_junction_embedding], dim=-1)
            intersection_len = []
            intersection_action, intersection_junction, max_intersection_len =intersection_history

            action = np.zeros([len(intersection_action), max_intersection_len[0]])
            junction = np.zeros([len(intersection_action), max_intersection_len[0]])
            for b in range(len(intersection_action)):
                is_len = len(intersection_action[b])
                intersection_len.append(is_len)
                action[b,:is_len] = intersection_action[b]
                junction[b,:is_len] = intersection_junction[b]
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
            junction = torch.tensor(junction, dtype=torch.int64).to(self.device)
            action_embedding = self.action_embed(action)
            junction_type_embedding = self.junction_type_embed(junction)
            intersection_history_feat = torch.concat([action_embedding, junction_type_embedding], dim=-1)
            # self-attention
            # padding
            intersection_history_feat_all = torch.concat([now_feat, intersection_history_feat.transpose(0,1)], dim=0)
            intersection_attention_input = self.intersection_history_projection_layer(intersection_history_feat_all)
            intersection_attention_input = self.intersection_history_dropout(intersection_attention_input)
            # intersection_history_feat = intersection_history_feat.transpose(0,1)
            intersection_len = torch.tensor(intersection_len, dtype=torch.int64)
            key_padding_mask = ~(torch.arange(intersection_attention_input.shape[0])[None, :] < (intersection_len[:, None]+1))
            key_padding_mask = key_padding_mask.to(self.device)

            attn, attn_weights = self.intersection_attention_layer(query=intersection_attention_input,
                                                           key=intersection_attention_input,
                                                           value=intersection_attention_input,
                                                           key_padding_mask=key_padding_mask)
            if self.opts.config.use_layer_norm:
                attn = self.layer_norm_intersection_attention(attn)

            attn_output = torch.concat([hidden_state, attn[0].unsqueeze(0)], dim=-1)
            return attn_output


    def _text_attention(self, trajectory_hidden_state, text_enc_outputs, text_enc_lengths):
        key_padding_mask = ~(torch.arange(text_enc_outputs.shape[0])[None, :] < text_enc_lengths[:, None])
        key_padding_mask = key_padding_mask.to(self.device)

        attn, attn_weights = self.text_attention_layer(query=trajectory_hidden_state,
                                                       key=text_enc_outputs,
                                                       value=text_enc_outputs,
                                                       key_padding_mask=key_padding_mask)
        if self.opts.config.use_layer_norm:
            attn = self.layer_norm_text_attention(attn)
        return attn, attn_weights

    def _visual_attention(self, text_attn, image_features):
        image_features = image_features.permute(1, 0, 2)
        attn, attn_weights = self.visual_attention_layer(query=text_attn,
                                                         key=image_features,
                                                         value=image_features)
        if self.opts.config.use_layer_norm:
            attn = self.layer_norm_visual_attention(attn)
        return attn
