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

dataset = 'map2seq_seen'
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
                # left or right + forward once cannot reach the target node
                # In this case, we cannot determine whether gt is left or right, so we add both left and right to the action sequence.
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
    Extract the current instruction's action phrase (node).
    """

    def forbidden_word_detection(node, idx, chunks):
        for word, tag in node:
            if word == "n't" or word == "look":
                return True
        # Check the previous 2 chunks for words like "after"
        IN_list = ['before', 'after', 'if', 'once', 'until', 'chance', 'where', 'want']
        detection_idx = idx - 1
        while detection_idx >= idx - 4 and detection_idx >= 0:
            detection_node = chunks[detection_idx]
            if isinstance(detection_node, Tree):
                for word, tag in detection_node:
                    if word in [',', '.', '!', 'and']:
                        # Ensure the checked chunk is in the same sentence as the current chunk
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
        # Determine if the verb chunk contains a direction; if not, try to add it
        if action and not direction:
            # Determine if node_next contains a direction
            direction_next = False
            if node_next is not None:
                for idx, item in enumerate(node_next):
                    if item[0] in direction_list:
                        direction_next = True
            if direction_next:
                # Add node_next to node
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
    # Prefer left-aligned matches in case of overlapping matches
    chunks = cp.parse(tagged)
    # print(chunks)
    action_pos = []
    for idx, node in enumerate(chunks):
        if isinstance(node, Tree):
            # Skip adverbs and "to"
            if node._label == 'VB' and not (len(node) == 1 and (node[0][1] == 'TO' or node[0][1] == 'RB')):
                # Check if the chunk contains an action word and detect the action
                node_next = chunks[idx+1] if idx < len(chunks)-1 else None
                if_action, importance_score = action_detection(node, node_next, data_idx, instr_idx)
                if not forbidden_word_detection(node, idx, chunks) and if_action:
                    # print(node)
                    clear_action = get_clear_action(node)
                    action_pos.append([clear_action, instr_idx, importance_score, ''])
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
    Output the navigation information for each node in the trajectory corresponding to the current instruction
    For each node, output:
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
        Check if the current trajectory node meets certain conditions
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
                        # Adjust heading if it exceeds 180
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
        Check if the current instruction node meets certain conditions
        """
        if detection_type in [0,2,3]:
            # Check for "left" or "right" with importance_score=0
            if instr_at[0] in ['left', 'right'] and instr_at[2] == 0:
                return True
            else:
                return False
        elif detection_type == 1:
            # Check for "left" or "right" with importance_score=-1 or 0
            if instr_at[0] in ['left', 'right']:
                return True
            else:
                return False

    def is_paired(traj_turn, instr_turn, traj_info):
        # Calculate the number of nodes that stay at the starting point
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
        # Output the split and paired results
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
    Divide the entire navigation into multiple stages and match the trajectory and instructions to the corresponding stages
    Returns:
    - `stage_instr_pair`: Stores the pairing of sub-instructions in `instr` with stages
	- `stage_traj_pair`: Stores the pairing of sub-actions in `traj` with stages
    """
    gt_action = ['forward', 'left', 'right', 'stop']
    # Determine how many stages the navigation has
    ## Determine how many actions the trajectory has and whether there are nodes after the last action
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
    ## Determine how many actions the instructions have and whether there are sentences after the last action
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

    # Start pairing
    if is_paired(traj_turn, instr_turn, traj_info):
        # Match successfully, start pairing
        pair_stages = len(traj_turn)+1 if (instr_remain and traj_remain) else len(traj_turn)
        stage_instr_pairs = []
        stage_traj_pairs = []
        traj_stage_start_idx = 0
        instr_stage_start_idx = 0
        for pair_idx in range(pair_stages):
            stage_traj_pair_item = []
            stage_instr_pair_item = []
            if pair_idx == len(traj_turn):
                # This is the end stage
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
                    # The current instruction and the previous one belong to the same stage
                    stage_instr_pair_item = stage_instr_pairs[-1]
                else:
                    for idx in range(instr_stage_start_idx, instr_turn[pair_idx][1]+1):
                        stage_instr_pair_item.append(idx)
                    instr_stage_start_idx = instr_turn[pair_idx][1] + 1
            stage_traj_pairs.append(stage_traj_pair_item)
            stage_instr_pairs.append(stage_instr_pair_item)
        if traj_stage_start_idx < len(traj_info[0]):
            # Not all trajectory actions are stored yet
            if traj_stage_start_idx == 0:
                # The entire navigation is one stage
                stage_traj_pair_item = []
            else:
                stage_traj_pair_item = stage_traj_pairs[-1]
            for idx in range(traj_stage_start_idx, len(traj_info[0])):
                stage_traj_pair_item.append(idx)
        if instr_stage_start_idx < len(instrs):
            # Not all instructions are stored yet
            if instr_stage_start_idx == 0:
                stage_instr_pair_item = []
            else:
                stage_instr_pair_item = stage_instr_pairs[-1]
            for idx in range(instr_stage_start_idx, len(instrs)):
                stage_instr_pair_item.append(idx)
        print_instr_traj(stage_traj_pairs, stage_instr_pairs, instrs, traj_info)
        return True, stage_traj_pairs, stage_instr_pairs
    else:
        # Matching failed
        print('--------------------------------------------------------')
        get_action_list(traj_info)
        print('--------------------------------------------------------')

        return False, [], []


def get_node_instr_pair(if_pair_success, stage_traj_pairs, stage_instr_pairs, traj_info, instrs, sample_info, data_idx):
    traj_action_len = len(traj_info[0])
    instr_len = len(instrs)
    node_list = sample_info['route_panoids']

    if not if_pair_success:
        # Build stage_traj_pairs and stage_instr_pairs
        stage_traj_pairs = [list(range(traj_action_len))]
        stage_instr_pairs = [list(range(instr_len))]

    # Start building node_instr_pair
    ## Build node_sub_pair
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

    ## Build node_instr_pair
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
        # last_instr_action_instr_id indicates the last importance=0 left/right action in which sentence
        last_instr_action_instr_id = 0
        for instr_idx, instr in enumerate(instrs):
            # print(instr)
            tagged_token = token_tagging(instr, i, instr_idx)
            for item in tagged_token:
                instr_action_info.append(item)
                if item[0] in ['left', 'right'] and item[2] == 0:
                    last_instr_action_instr_id = item[1]

        # 4 types of pairing strategies
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
        node_instr_pair_all[data['id']] = {}
        node_instr_pair_all[data['id']]['node_instr_pair'] = node_instr_pair
        node_instr_pair_all[data['id']]['confidence_score'] = confidence_score

    success = sum(if_pair_success_all)
    print(success/len(if_pair_success_all))

    import pickle as pkl
    pkl.dump(node_instr_pair_all, open(f'/root/VLN/code/thl_vln/result_visualization/tmp_files/{dataset}_node_instr_pair_all.pkl', 'wb'))



if __name__ == "__main__":
    main()