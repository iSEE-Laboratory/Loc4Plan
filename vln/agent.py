import numpy
import torch
from torch import nn
import numpy as np
from utils import padding_idx, heading_normalization


class BaseAgent:
    def __init__(self, opts, env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.env = env

    @staticmethod
    def _pad_seq(seqs):
        max_len = max([len(seq) for seq in seqs])
        seq_tensor = list()
        seq_lengths = list()
        for seq in seqs:
            instr_encoding_len = len(seq)
            seq = seq + [padding_idx] * (max_len - instr_encoding_len)
            seq_tensor.append(seq)
            seq_lengths.append(instr_encoding_len)
        seq_tensor = np.array(seq_tensor, dtype=np.int64)
        seq_lengths = np.array(seq_lengths, dtype=np.int64)
        return seq_tensor, seq_lengths

    def get_batch(self):
        """ Extract instructions from a list of observations and calculate corresponding seq lengths. """
        seqs = [list(item['instr_encoding']) for item in self.env.batch]

        seq_tensor, seq_lengths = self._pad_seq(seqs)

        seq_tensor = torch.from_numpy(seq_tensor).to(self.device)
        seq_lengths = torch.from_numpy(seq_lengths)

        # visualization
        tokens = [item['visualization_token'] for item in self.env.batch]
        instrs = [item['navigation_text'] for item in self.env.batch]
        gt_nodes = [item['route_panoids'] for item in self.env.batch]
        start_headings = [item['start_heading'] for item in self.env.batch]
        return seq_tensor, seq_lengths, [tokens, instrs, gt_nodes, start_headings]


class OutdoorVlnAgent(BaseAgent):
    def __init__(self, opts, env, encoder, model):
        super(OutdoorVlnAgent, self).__init__(opts, env)
        self.instr_encoder = encoder
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=4)
        self.criterion_progress = nn.MSELoss()


    def rollout(self, is_test):
        trajs = self.env.reset()  # a batch of the first panoid for each route_panoids
        vlz_att_weights = []
        traj_nodes = []
        agent_actions = []
        headings = []
        batch_size = len(self.env.batch)

        confidence_weight = [1, 0.7, 0.5, 0.2, 0]
        node_instr_confidence_score = self.env.get_node_instr_confidence_score()
        weight = [confidence_weight[score] for score in node_instr_confidence_score]
        weight = torch.tensor(weight).to(self.device)

        seq, seq_lengths, vlz_info_sub= self.get_batch()
        (text_enc_outputs, text_enc_lengths), (first_ht, first_ct) = self.instr_encoder(seq, seq_lengths)  # LSTM encoded hidden states for instructions, [batch_size, 1, 256]
        h_t = first_ht
        c_t = first_ct

        ended = np.zeros(batch_size, dtype=np.bool)
        ended = torch.from_numpy(ended).to(self.device)
        h2_t = None
        c2_t = None

        a = torch.ones(batch_size, 1, dtype=torch.int64, device=self.device)
        heading_changes = torch.zeros(batch_size, 3, 1, dtype=torch.float32, device=self.device)
        t = torch.tensor([-1], dtype=torch.int64, device=self.device)
        num_act_nav = [batch_size]
        loss = 0
        total_steps = [0]
        block_progress_all = None
        intersection_action = []
        intersection_junction = []
        max_intersection_len = [0]
        loss_list = {"action_loss":0, "progress_loss":0, "subInstr_loss":0}
        max_instr_len = self.env.get_max_instr_len()
        subInstr_feat = self.env.get_subInstr_feat(text_enc_outputs, max_instr_len)
        token_instr_id = self.env.get_token_instr_id()

        action_heading_change = self.env.env.get_action_heading_change(self.opts.config.heading_change_noise, is_test, batch_size)
        action_heading_change = np.asarray(action_heading_change, dtype=np.float32)
        action_heading_change = torch.from_numpy(action_heading_change).to(self.device)

        progress_all = []

        for step in range(self.opts.max_route_len):
            image_features = self.env.get_imgs()
            junction_types = self.env.get_junction_type()
            block_progress_all = self.env.get_block_progress(block_progress_all)
            block_progress = [2*(0.5 - block_progress_all['step'][i]/block_progress_all['all'][i]) for i in range(len(block_progress_all['all']))]
            progress_all.append(block_progress)

            instr_mask = self.env.get_instr_mask()
            # token_instr_pair = self.env.get_token_instr_pair()
            # max_instr_len = self.env.get_max_instr_len()
            # node_instr_pair = self.env.get_node_instr_pair()
            # node_pose = self.env.get_node_pose()
            token_instr_pair, max_instr_len, node_instr_pair, node_pose, instr_lenghs, route_id_all = self.env.get_sub_instr_info()

            # print(f"{block_progress['step'][0]}/{block_progress['all'][0]}")
            t = t + 1

            # heading_normalization
            heading_change_1 = heading_changes[:, -1, :].unsqueeze(dim=1)
            heading_change_1 = heading_normalization(heading_change_1, self.opts.config.heading_change_noise, is_test)
            heading_change_3 = heading_changes.sum(dim=1).unsqueeze(dim=1)
            heading_change_3 = heading_normalization(heading_change_3, self.opts.config.heading_change_noise, is_test)

            policy_output, (h_t, c_t), (h2_t, c2_t), vlz_att_weight, progress_score, node_instr_score = self.model(text_enc_outputs,
                                                                 text_enc_lengths,
                                                                 token_instr_pair,
                                                                 token_instr_id,
                                                                 max_instr_len,
                                                                 instr_lenghs,
                                                                 subInstr_feat,
                                                                 image_features,
                                                                 a,
                                                                 junction_types,
                                                                 [heading_change_1, heading_change_3, action_heading_change],
                                                                 block_progress,
                                                                 [intersection_action, intersection_junction, max_intersection_len],
                                                                 h_t,
                                                                 c_t,
                                                                 t,
                                                                 h2_t=h2_t,
                                                                 c2_t=c2_t,
                                                                 step = step,
                                                                 instr_mask = instr_mask
                                                                 )
            if is_test:
                a, heading_changes_new, action_heading_change, traj_state = self.env.action_select(policy_output, ended, num_act_nav, trajs, total_steps, self.opts.config.heading_change_noise)
                heading_changes = torch.concat([heading_changes[:, -2:, :], heading_changes_new], dim=1)
                vlz_att_weights.append(vlz_att_weight[0])
                traj_nodes.append(traj_state[0])
                agent_actions.append(a)
                headings.append(torch.Tensor(traj_state[1]).unsqueeze(dim=1))
            else:
                gold_actions = self.env.get_gt_action()
                target_ = gold_actions.masked_fill(ended, value=torch.tensor(4))
                loss += self.criterion(policy_output, target_) * num_act_nav[0]
                loss_list['action_loss'] += (self.criterion(policy_output, target_) * num_act_nav[0]).item()
                # progress_score
                if self.opts.config.use_progress_score:
                    progress_score_prd = progress_score.squeeze(0).squeeze(-1)[~ended]
                    progress_gt = torch.Tensor(block_progress).to(self.device)[~ended]
                    loss += self.criterion_progress(progress_score_prd, progress_gt)
                    loss_list['progress_loss'] += (self.criterion_progress(progress_score_prd, progress_gt)).item()

                if self.opts.config.use_subInstruction:
                    # subInstruction
                    subInstr_loss = 0
                    for instr_idx in range(max_instr_len):
                        sub_instr_pos = [node_instr_pair[i][node_pose[i]] for i in range(len(node_instr_pair))]
                        sub_instr_gt = [(instr_idx in item) for item in sub_instr_pos]
                        sub_instr_gt = torch.tensor(sub_instr_gt, dtype=torch.float32).to(self.device)
                        # node_instr_score = nn.Sigmoid()(node_instr_score)
                        instr_mask = torch.BoolTensor([instr_idx <= len-1 for len in instr_lenghs]).to(self.device)
                        ended_mask = ~ended
                        batch_mask = instr_mask & ended_mask
                        masked_weight = weight[batch_mask]
                        if masked_weight.shape[0] != 0:
                            criterion_stage = nn.BCELoss(weight=masked_weight)
                            subInstr_loss += criterion_stage(node_instr_score[:,instr_idx,0][batch_mask], sub_instr_gt[batch_mask])
                    subInstr_loss = subInstr_loss / max_instr_len
                    loss += subInstr_loss
                    loss_list['subInstr_loss'] += subInstr_loss

                heading_changes_new, action_heading_change = self.env.env.action_step(gold_actions, ended, num_act_nav, trajs, total_steps)
                heading_changes = torch.concat([heading_changes[:, -2:, :], heading_changes_new], dim=1)
                self.update_intersection_history(intersection_action, intersection_junction, max_intersection_len, gold_actions, junction_types)
                a = gold_actions.unsqueeze(1)
            if not num_act_nav[0]:
                break

        progress_all = numpy.array(progress_all)

        loss /= total_steps[0]
        loss_list['action_loss']/= total_steps[0]
        loss_list['progress_loss']/= total_steps[0]
        loss_list['subInstr_loss']/= total_steps[0]
        if is_test:
            vlz_att_weights = torch.concat(vlz_att_weights, dim=1)
            # traj_nodes
            traj_nodes_2 = []
            for i in range(len(traj_nodes[0])):
                traj_nodes_list = []
                for j in range(len(traj_nodes)):
                    traj_nodes_list.append(traj_nodes[j][i])
                traj_nodes_2.append(traj_nodes_list)
            agent_actions = torch.concat(agent_actions, dim=1)
            headings = torch.concat(headings, dim=1)
            vlz_info = {
                'weights': vlz_att_weights,
                'trajs': traj_nodes_2,
                'actions': agent_actions,
                'headings': headings,
                'tokens': vlz_info_sub[0],
                'instrs':vlz_info_sub[1],
                'gt_nodes': vlz_info_sub[2],
                'start_headings': vlz_info_sub[3]
            }
            return None, trajs, vlz_info
        else:
            return loss, loss_list, None, None


    def update_intersection_history(self, intersection_action, intersection_junction, max_intersection_len, actions, junction_types):
        for b in range(actions.shape[0]):
            if b >= len(intersection_action):
                intersection_action.append([])
                intersection_junction.append([])
            if junction_types[b] != 0:
                intersection_action[b].append(actions[b].item())
                intersection_junction[b].append(junction_types[b].item())
                if max_intersection_len[0] < len(intersection_action[b]):
                    max_intersection_len[0] = len(intersection_action[b])
