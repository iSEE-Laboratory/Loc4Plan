from copy import deepcopy

import torch
import warnings
import sys
import os
import json
from vln.graph_loader import GraphLoader
import nltk
from nltk import Tree

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
warnings.filterwarnings('ignore')
sys.path.append("/root/VLN/code/thl_vln")
sys.path.append("/root/VLN/code/thl_vln/vln")

dataset = 'touchdown_unseen'
dataset_dir = f'/root/VLN/code/thl_vln/datasets/{dataset}'

graph = GraphLoader(f'/root/VLN/code/map2seq_vln/datasets/{dataset}').construct_graph()

# nltk.download()
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def load_datasets(splits, opts=None):
    data = []
    for split in splits:
        assert split in ['train', 'test', 'dev']
        with open('%s/data/%s.json' % (dataset_dir, split)) as f:
            for line in f:
                item = dict(json.loads(line))
                item["navigation_text"] = item["navigation_text"].lower()
                data.append(item)
    return data

def trajs_info(route_panoids, start_heading):
    from vln.base_navigator import BaseNavigator

    gt_actions = []
    gt_junctions = []
    gt_traj_all = []
    heading_change = []

    # gt_action
    nav = BaseNavigator(f'/root/VLN/code/thl_vln/datasets/{dataset}')
    nav.graph_state = (route_panoids[0], start_heading)
    nav.initial_pano_id = route_panoids[0]

    action_list = ['forward', 'left', 'right']

    for step in range(100):
        find_action = False
        if_start = False
        node_idx = route_panoids.index(nav.graph_state[0])
        node_next = route_panoids[node_idx + 1]
        action = []

        # forward
        tmp_node, tmp_heading = nav._get_next_graph_state(nav.graph_state, action_list[0])
        if tmp_node == nav.graph_state[0]:
            # step 0
            if_start = True
            action.append(0)
            find_action = True
        elif tmp_node == node_next:
            action.append(0)
            find_action = True
        else:
            # left/right
            tmp_node, tmp_heading = nav._get_next_graph_state(nav.graph_state, action_list[1])
            tmp = nav._get_next_graph_state([tmp_node, tmp_heading], action_list[0])
            if tmp[0] == node_next:
                find_action = True
                action.append(1)

            tmp_node, tmp_heading = nav._get_next_graph_state(nav.graph_state, action_list[2])
            tmp = nav._get_next_graph_state([tmp_node, tmp_heading], action_list[0])
            if tmp[0] == node_next:
                find_action = True
                action.append(2)

            if not find_action:
                # left 或者 right +forward 一次都到不了目标节点
                # 说明可能要多转几次。这种情况下我们无法确定 gt 是 left 还是 right，就把 left、right 都加入动作序列。
                action.append(1)
                action.append(2)
                find_action = True

        assert find_action
        gt_actions.append(action)
        info = {}
        info['step'] = step
        info['action'] = action
        info['start_state'] = nav.graph_state
        neibors = graph.nodes[nav.graph_state[0]].neighbors
        gt_junctions.append(len(neibors))

        # step
        heading_1 = nav.graph_state[1]
        if len(action) == 1:
            nav.step(action_list[action[0]])
        else:
            tmp = nav.graph_state
            while tmp[0] != node_next:
                nav.step(action_list[action[0]])
                tmp = nav._get_next_graph_state(nav.graph_state, action_list[0])
        heading_2 = nav.graph_state[1]
        # print(f'{nav.graph_state}')
        info['end_state'] = nav.graph_state
        gt_traj_all.append(info)
        heading_change.append(heading_2-heading_1)

        if nav.graph_state[0] == route_panoids[-1]:
            break
    return gt_actions, gt_junctions, gt_traj_all, heading_change


def token_tagging(str, data_idx, instr_idx):
    """
    提取出当前指令的动作短语(node)。
    """

    def forbidden_word_detection(node, idx, chunks):
        for word, tag in node:
            if word == "n't" or word == "look":
                return True
        # print(111)
        # 往前查2个词块，有没有after等介词
        IN_list = ['before', 'after', 'if', 'once', 'until', 'chance', 'where', 'want']
        detection_idx = idx - 1
        while detection_idx >= idx - 4 and detection_idx >= 0:
            detection_node = chunks[detection_idx]
            if isinstance(detection_node, Tree):
                for word, tag in detection_node:
                    if word in [',', '.', '!', 'and']:
                        # 保证检查的词块和当前词块在同一个短句中
                        return False
                    elif word in IN_list:
                        return True
            else:
                if detection_node[0] in [',', '.', '!', 'and']:
                    return False
                elif detection_node[0] in IN_list:
                    return True
            detection_idx -= 1
        return False

    def action_detection(node, node_next, data_idx, instr_idx):
        action_list = ["go", "turn", "take", "walk", "move", "make"]
        direction_list = ["left", "right", "straight", "forward"]
        # VB_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

        importance_score = 0

        direction = False
        action = False
        for word, tag in node:
            if word in action_list:
                action = True
            if word in direction_list:
                direction = True
            # if tag in VB_list:
            #     VB = True
        if direction and not action:
            return [True, -1]
        # 确定 动词node 中是否包含方向，如果没有，尝试补上
        if action and not direction:
            # 确定 node_next 中有没有方向
            direction_next = False
            if node_next is not None:
                for idx, item in enumerate(node_next):
                    if item[0] in direction_list:
                        direction_next = True
            if direction_next:
                # 把 node_next 加入 node
                for item in node_next:
                    node.append(item)
                return [True, 0]
            else:
                return [False, 0]
        elif action and direction:
            return [True, 0]
        return [False, 0]

    direction_list = ["left", "right", "straight", "forward"]
    action_list = ["go", "turn", "take", "walk", "move", "make", "look"]


    words = nltk.word_tokenize(str)
    tagged = nltk.pos_tag(words)
    # 把被错认为NP的动作词块修正为VB
    for idx, word in enumerate(tagged):
        if word[0] in action_list:
            word = list(word)
            word[1] = 'VB'
            tagged[idx] = tuple(word)
        elif word[0] in direction_list:
            if_and  = False
            if_and  = if_and or (idx + 1 < len(tagged) and tagged[idx+1][0] == 'and')
            if_and  = if_and or (idx - 1 >= 0 and tagged[idx-1][0] == 'and')
            if_the = False or (idx - 1 >= 0 and tagged[idx-1][0] in ['the', 'your', 'a', 'another'])
            if if_and and not if_the:
            # (tagged[idx+1][0] == 'and' or tagged[idx-1][0] == 'and') and tagged[idx-1][0] != 'the':
                word = list(word)
                word[1] = 'VB'
                tagged[idx] = tuple(word)
    grammar = r"""
            NP: {<DT|PP\$|PRP\$>?<JJ>*<NN|NNS>}   # chunk determiner/possessive, adjectives and noun
            {<NNP>+}                # chunk sequences of proper nouns
            VB: {<VBZ?|VBP?|VB?|VBD?|TO?|VBN?|JJ?|VBG?>*<RB?|RP?>*<VBZ?|VBP?|VB?|VBD?|TO?|VBN?|JJ?|VBG?>*<RB?|RP?>*} 
        """
    cp = nltk.RegexpParser(grammar)
    # 重叠匹配则按照左边匹配优先
    chunks = cp.parse(tagged)
    # print(chunks)
    action_pos = []
    for idx, node in enumerate(chunks):
        if isinstance(node, Tree):
            # # 词块包含单词 turn，且出现在句子开头
            # if instr_idx == 0 and idx == 0 and node[0][0] in ['turn', 'orient']:
            #     print(node)
            # 排除单纯的副词、to
            if node._label == 'VB' and not (len(node) == 1 and (node[0][1] == 'TO' or node[0][1] == 'RB')):
                # 排除不想看到的词(don't look)，检测动作短语
                node_next = chunks[idx+1] if idx < len(chunks)-1 else None
                if_action, importance_score = action_detection(node, node_next, data_idx, instr_idx)
                if not forbidden_word_detection(node, idx, chunks) and if_action:
                    # print(node)
                    clear_action = get_clear_action(node)
                    action_pos.append([clear_action, instr_idx, importance_score, ''])
                # 保证最终的动词短语里包含方向信息
        # elif node[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        #     print(node)
        #     clear_action = get_clear_action(node)
        #     action_pos.append([clear_action, instr_idx])
    return action_pos


def get_clear_action(node):
    clear_action = None
    direction_list = ["left", "right", "straight", "forward"]

    for text, _ in node:
        if text in direction_list:
            clear_action = text
    assert clear_action is not None
    return clear_action


def get_action_list(info):
    """
    输出当前指令对应轨迹的每个节点的导航情况
    对于每个节点，输出：
    gt_action(heading_change)(junctions)
    """
    gt_actions, gt_junctions, gt_traj_all, heading_change = info
    for i in range(len(gt_actions)):
        if gt_junctions[i] == 2:
            print(f'{gt_actions[i]}({int(heading_change[i])})  ', end='')
        else:
            print(f'{gt_actions[i]}({int(heading_change[i])})(*{gt_junctions[i]}*)  ', end='')
        if gt_actions[i][0] != 0:
            print('')
    print('')

def split_pair(instr_action_info, traj_info, instrs, data_idx, detection_type = 0):
    def is_need_traj_action(traj_info, traj_idx, detection_type = 0):
        """
        检测当前 traj节点是否是我们需要的“满足一定条件的”动作节点
        """
        traj_actions, traj_junctions, traj_all, traj_heading_change = traj_info
        traj_at = traj_actions[traj_idx]
        past_traj_at = traj_actions[traj_idx-1] if traj_idx > 0 else [1,2]
        if detection_type in [0,1]:
            if traj_at[0] in [1, 2]:
                gt_action = ['forward', 'left', 'right', 'stop']
                traj_turn_item = [gt_action[item] for item in traj_at]
                return True, traj_turn_item
            else:
                return False, []
        elif detection_type == 2:
            if traj_at[0] == 0 and past_traj_at not in [1, 2] and traj_junctions[traj_idx] > 2:
                traj_hc = traj_heading_change[traj_idx]
                if abs(traj_hc) > 37 and abs(traj_hc) < 115:
                    true_action = 'left' if traj_hc<0 else 'right'
                    return True, [true_action]
            elif traj_at[0] in [1, 2]:
                gt_action = ['forward', 'left', 'right', 'stop']
                traj_turn_item = [gt_action[item] for item in traj_at]
                return True, traj_turn_item
            else:
                return False, []
        else:
            if traj_at[0] == 0 and traj_idx > 2:
                past_traj_at_1 = traj_actions[traj_idx - 1]
                past_traj_at_2 = traj_actions[traj_idx - 2]
                # past_traj_hc = traj_heading_change[traj_idx - 1] + traj_heading_change[traj_idx - 2] + traj_heading_change[traj_idx]
                past_traj_hc = 0
                for i in [traj_idx - 1, traj_idx - 2, traj_idx]:
                    heading = traj_heading_change[i]
                    if abs(heading) > 180:
                        # 需要修改
                        heading = heading + 360 if heading < 0 else heading - 360
                    past_traj_hc += heading
                if 0 in past_traj_at_1 and 0 in past_traj_at_2 and abs(past_traj_hc)>57:
                    true_action = 'left' if past_traj_hc < 0 else 'right'
                    traj_heading_change[traj_idx - 1] = 0
                    traj_heading_change[traj_idx - 2] = 0
                    traj_heading_change[traj_idx] = 0
                    return True, [true_action]
            elif traj_at[0] in [1, 2]:
                gt_action = ['forward', 'left', 'right', 'stop']
                traj_turn_item = [gt_action[item] for item in traj_at]
                return True, traj_turn_item
            else:
                return False, []
        return False, []

    def is_need_instr_action(instr_at, detection_type = 0):
        """
        检测当前 instr 节点是否是我们需要的“满足一定条件的”动作节点
        """
        if detection_type in [0,2,3]:
            # 检测 left/right 且 inportance_score=0的节点
            if instr_at[0] in ['left', 'right'] and instr_at[2] == 0:
                return True
            else:
                return False
        elif detection_type == 1:
            # 检测 left/right 且 importance_score=-1/0的节点
            if instr_at[0] in ['left', 'right']:
                return True
            else:
                return False

    def is_paired(traj_turn, instr_turn, traj_info):
        # 计算呆在起点的节点数
        traj_actions, traj_junctions, traj_all, traj_heading_change = traj_info
        start_node_idx = traj_all[0]['start_state'][0]
        start_step = len(traj_actions)
        for i in range(len(traj_actions)):
            if traj_all[i]['start_state'][0] == start_node_idx and (
                    i == len(traj_actions) - 1 or traj_all[i + 1]['start_state'][0] != start_node_idx):
                start_step = i - 1
                break

        if len(traj_turn) > 0 and traj_turn[0][1] <= start_step:
            traj_turn.pop(0)

        if len(traj_turn) != len(instr_turn):
            return False
        for idx in range(len(traj_turn)):
            if not instr_turn[idx][0] in traj_turn[idx][0]:
                return False
        return True

    def print_instr_traj(stage_traj_pairs, stage_instr_pairs, instrs, traj_info):
        # 输出拆分配对结果
        gt_actions, gt_junctions, gt_traj_all, heading_change = traj_info
        for i in range(len(stage_traj_pairs)):
            for instr_idx in stage_instr_pairs[i]:
                print(instrs[instr_idx], end='')
                print('.', end='')
            print()
            for traj_idx in stage_traj_pairs[i]:
                if gt_junctions[traj_idx] == 2:
                    print(f'{gt_actions[traj_idx]}({int(heading_change[traj_idx])})  ', end='')
                else:
                    print(f'{gt_actions[traj_idx]}({int(heading_change[traj_idx])})(*{gt_junctions[traj_idx]}*)  ', end='')
            print()
        print()

    """
    把整个导航分为多个阶段，然后把 traj 和 instr 的各个阶段匹配到对应的阶段
    返回：
    - `stage_instr_pair`: 存储 instr 中子指令和 stage 的配对关系
	- `stage_traj_pair`: 存储 traj 中子动作和 stage 的配对关系
    """
    gt_action = ['forward', 'left', 'right', 'stop']
    # 确定该样本对应的导航有几个阶段
    ## 确定 traj 有几个动作，最后一个动作后还有没有节点
    traj_actions, traj_junctions, traj_all, traj_heading_change = traj_info
    traj_remain = False
    traj_turn = []
    # for traj_idx, traj_at in enumerate(traj_actions):
    #     if traj_at[0] in [1,2]:
    #         traj_turn_item = [gt_action[item] for item in traj_at]
    #         traj_turn.append([traj_turn_item, traj_idx])
    #         traj_remain = False
    #     else:
    #         traj_remain = True
    for traj_idx in range(len(traj_actions)):
        traj_at = traj_actions[traj_idx]
        is_need_traj_ac, traj_turn_item = is_need_traj_action(traj_info, traj_idx, detection_type)
        if is_need_traj_ac:
            # traj_turn_item = [gt_action[item] for item in traj_at]
            traj_turn.append([traj_turn_item, traj_idx])
            traj_remain = False
        else:
            traj_remain = True
    ## 确定 instr 有几个动作，最后一个动作所属句子后，还有没有其他句子
    instr_turn = []
    instr_last_action_instr_id = -1
    for instr_at in instr_action_info:
        if is_need_instr_action(instr_at, detection_type):
            instr_turn.append([instr_at[0], instr_at[1]])
            instr_last_action_instr_id = instr_at[1]
    if instr_last_action_instr_id < len(instrs)-1:
        instr_remain = True
    else:
        instr_remain = False

    # 开始匹配
    if is_paired(traj_turn, instr_turn, traj_info):
        # 匹配成功，开始配对
        pair_stages = len(traj_turn)+1 if (instr_remain and traj_remain) else len(traj_turn)
        stage_instr_pairs = []
        stage_traj_pairs = []
        traj_stage_start_idx = 0
        instr_stage_start_idx = 0
        for pair_idx in range(pair_stages):
            stage_traj_pair_item = []
            stage_instr_pair_item = []
            if pair_idx == len(traj_turn):
                # 说明这是end阶段
                for idx in range(traj_stage_start_idx, len(traj_info[0])):
                    stage_traj_pair_item.append(idx)
                traj_stage_start_idx = len(traj_info[0])

                for idx in range(instr_stage_start_idx, len(instrs)):
                    stage_instr_pair_item.append(idx)
                instr_stage_start_idx = len(instrs)
            else:
                for idx in range(traj_stage_start_idx, traj_turn[pair_idx][1]+1):
                    stage_traj_pair_item.append(idx)
                traj_stage_start_idx = traj_turn[pair_idx][1] + 1

                if instr_stage_start_idx == instr_turn[pair_idx][1]+1:
                    # 当前instr和上一个instr属于同一个stage
                    stage_instr_pair_item = stage_instr_pairs[-1]
                else:
                    for idx in range(instr_stage_start_idx, instr_turn[pair_idx][1]+1):
                        stage_instr_pair_item.append(idx)
                    instr_stage_start_idx = instr_turn[pair_idx][1] + 1
            stage_traj_pairs.append(stage_traj_pair_item)
            stage_instr_pairs.append(stage_instr_pair_item)
        if traj_stage_start_idx < len(traj_info[0]):
            # 还没存储完
            if traj_stage_start_idx == 0:
                # 整个导航只有一个阶段
                stage_traj_pair_item = []
            else:
                stage_traj_pair_item = stage_traj_pairs[-1]
            for idx in range(traj_stage_start_idx, len(traj_info[0])):
                stage_traj_pair_item.append(idx)
        if instr_stage_start_idx < len(instrs):
            # 还没存储完
            if instr_stage_start_idx == 0:
                stage_instr_pair_item = []
            else:
                stage_instr_pair_item = stage_instr_pairs[-1]
            for idx in range(instr_stage_start_idx, len(instrs)):
                stage_instr_pair_item.append(idx)
        print_instr_traj(stage_traj_pairs, stage_instr_pairs, instrs, traj_info)
        return True, stage_traj_pairs, stage_instr_pairs
    else:
        # 匹配失败
        print('--------------------------------------------------------')
        get_action_list(traj_info)
        print('--------------------------------------------------------')

        return False, [], []


def get_node_instr_pair(if_pair_success, stage_traj_pairs, stage_instr_pairs, traj_info, instrs, sample_info, data_idx):
    traj_action_len = len(traj_info[0])
    instr_len = len(instrs)
    node_list = sample_info['route_panoids']

    if not if_pair_success:
        # 构建 stage_traj_pairs, stage_instr_pairs
        stage_traj_pairs = [list(range(traj_action_len))]
        stage_instr_pairs = [list(range(instr_len))]

    # 开始构建 node_instr_pair
    ## 构建 node_sub_pair
    node_stage_pair = []
    for stage_idx, stage_traj_item in enumerate(stage_traj_pairs):
        for traj_action_idx in stage_traj_item:
            node_idx = node_list.index(traj_info[2][traj_action_idx]['start_state'][0])
            # node_stage_pair[node_list.index(node_idx)].append(stage_idx)
            if node_idx >= len(node_stage_pair):
                node_stage_pair.append([stage_idx])
            else:
                if stage_idx not in node_stage_pair[node_idx]:
                    node_stage_pair[node_idx].append(stage_idx)
    node_stage_pair.append(node_stage_pair[-1])
    assert len(node_stage_pair) == len(node_list)

    ## 构建 node_instr_pair
    node_instr_pair = []
    for node_idx, node_stage_item in enumerate(node_stage_pair):
        for stage_idx in node_stage_item:
            if len(node_instr_pair) == node_idx:
                node_instr_pair.append(deepcopy(stage_instr_pairs[stage_idx]))
            else:
                for instr_idx in stage_instr_pairs[stage_idx]:
                    if instr_idx not in node_instr_pair[node_idx]:
                        node_instr_pair[node_idx].append(instr_idx)
    return node_instr_pair


def main():
    data_all = load_datasets(['train', 'test', 'dev'])

    if_pair_success_all = []
    node_instr_pair_all = {}

    for i in range(len(data_all)):
        data = data_all[i]
        traj_info = trajs_info(data['route_panoids'], data['start_heading'])
        # print("================trajs")
        # get_action_list(info)
        # print('================instructions')
        # print(data['navigation_text'])
        instrs = data['navigation_text'].split('.')
        if instrs[-1] == '':
            instrs.pop()

        instr_action_info = []
        # last_instr_action_instr_id 指示最后一个importance=0的左右动作在哪一句
        last_instr_action_instr_id = 0
        for instr_idx, instr in enumerate(instrs):
            # print(instr)
            tagged_token = token_tagging(instr, i, instr_idx)
            for item in tagged_token:
                instr_action_info.append(item)
                if item[0] in ['left', 'right'] and item[2] == 0:
                    last_instr_action_instr_id = item[1]

        # 3种配对策略，详见obsidian
        if_pair_success_final = False
        confidence_score = 4
        detection_type = [0,1,2,3]
        for type in detection_type:
            if_pair_success, stage_traj_pairs, stage_instr_pairs= split_pair(instr_action_info, traj_info, instrs, i, type)
            if if_pair_success:
                if_pair_success_final = True
                confidence_score = type
                break
        if_pair_success_all.append(if_pair_success_final)
        node_instr_pair = get_node_instr_pair(if_pair_success, stage_traj_pairs, stage_instr_pairs, traj_info, instrs, data, i)
        node_instr_pair_all[data['route_id']] = {}
        node_instr_pair_all[data['route_id']]['node_instr_pair'] = node_instr_pair
        node_instr_pair_all[data['route_id']]['confidence_score'] = confidence_score

    success = sum(if_pair_success_all)
    print(success/len(if_pair_success_all))

    import pickle as pkl
    pkl.dump(node_instr_pair_all, open(f'/root/VLN/code/thl_vln/result_visualization/tmp_files/{dataset}_node_instr_pair_all.pkl', 'wb'))



if __name__ == "__main__":
    main()
