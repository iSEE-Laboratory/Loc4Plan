import argparse
import os
import time

import torch
import numpy as np
import wandb

from utils import setup_seed, get_config, load_tokenizer, get_file_path
from utils import resume_training, set_tb_logger, save_checkpoint
from env import load_features, OutdoorVlnBatch
from agent import OutdoorVlnAgent
from trainer import OutdoorVlnTrainer
from instructions_encoder import InstructionEncoder
from model import ORAR
import os
import logging
from transformers import BertTokenizer

import warnings
warnings.filterwarnings("ignore")


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)


parser = argparse.ArgumentParser(description='PyTorch for Outdoor VLN on Touchdown and map2seq')
parser.add_argument('--config', default=None, type=str, help='Path to config file')
parser.add_argument('--dataset', default='touchdown_seen', type=str, choices=['touchdown_seen', 'touchdown_unseen', 'map2seq_seen', 'map2seq_unseen', 'merged_seen', 'merged_unseen'])
parser.add_argument('--img_feat_dir', default='', type=str, help='Path to pre-cached image features dir.')
parser.add_argument('--resume', default='', type=str, choices=['latest', 'TC_best', 'SPD_best'])
parser.add_argument('--store_ckpt_every_epoch', default=False, type=bool)
parser.add_argument('--test', default=False, action="store_true", help='No training. Resume from a model and run testing.')
parser.add_argument('--oracle', default='', type=str, help='e.g.: "rs". r=orientation; i=intersections;s=stop')
parser.add_argument('--seed', default=1234, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--eval_every_epochs', default=1, type=int, help='How often do we eval the trained model.')
parser.add_argument('--CLS', default=False, type=bool, help='Calculate CLS when evaluating.')
parser.add_argument('--DTW', default=False, type=bool, help='calculate DTW when evaluating.')
parser.add_argument('--max_route_len', default=55, type=int, help='Max trajectory length.')
parser.add_argument('--exp_name', default='experiments', type=str, help='Name of the experiment. It decides where to store samples and models')

parser.add_argument('--wandb', default=True, action='store_false', help='false means we didnt store output to wandb.')
parser.add_argument('--log', default=True, action='store_false', help='false means we didnt store output to log file.')
parser.add_argument('--start_time', default=None,help='start time of program running.')
parser.add_argument('--note', default=None,help='.')
parser.add_argument('--vlz_info_save', default=True, action='store_false',help='.')
parser.add_argument('--node_instr_pair_path', default='/root/VLN/code/thl_vln/instruction_preprocessing/tmp_files/', type=str, help='')
parser.add_argument('--service_id', default=667, type=int)

opts = parser.parse_args()

seed = opts.seed
setup_seed(seed)




def main(opts):

    if type(opts.config) == str:
        opts.config = get_config(opts.config, opts.oracle)
    print(opts.config)


    opts.dataset_dir = os.path.join('datasets', opts.dataset)
    opts.output_dir = os.path.join('outputs', opts.dataset, opts.exp_name)
    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, 'weights'), exist_ok=True)
    # os.makedirs(os.path.join(opts.output_dir, 'evaluation'), exist_ok=True)
    # os.makedirs(os.path.join(opts.output_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, 'log'), exist_ok=True)
    # opts.node_instr_pair_path = opts.node_instr_pair_path + f'{opts.dataset}_node_instr_pair_all.pkl'
    opts.node_instr_pair_path = get_file_path(opts.service_id, 'node_instr_pair', opts)

    if opts.config.tokenizer == 'spm':
        tokenizer = load_tokenizer(opts)
    else:
        BERT_TOKENIZER_DIR = '/root/VLN/models/huggingface/bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_DIR)

    image_features = None
    if opts.config.use_image_features:
        image_features = load_features(opts.img_feat_dir, features_name=opts.config.use_image_features, service_id = opts.service_id)

    if opts.test:
        main_test(opts, image_features, tokenizer)
    else:
        main_train(opts, image_features, tokenizer)


def main_train(opts, image_features, tokenizer):
    # iterations = opts.config.iterations
    iterations = opts.config.get('iterations', 10)
    val_tcs_from_spd = list()
    test_tcs_from_spd = list()
    best_epochs_from_spd = list()
    for i in range(iterations):
        # set start time
        current_time = time.time() + 16 * 3600
        current_time = time.strftime('%Y-%m-%d__%H-%M-%S', time.localtime(current_time))
        opts.start_time = current_time
        if opts.log:
            # log
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(opts.output_dir, 'log', f'{opts.start_time}_train.log'))
            logger.addHandler(file_handler)
            # stream_handler = logging.StreamHandler()
            # logger.addHandler(stream_handler)
            logger.info(opts)
        else:
            logger = None

        if opts.wandb:
            # wandb
            if opts.note is None:
                    note = None
            else:
                note = opts.note
            wandb.init(
                # set the wandb project where this run will be logged
                project="thl-vln",
                name=f"{opts.exp_name}({i})",
                # track hyperparameters and run metadata
                config={
                    "opts": opts,
                    "note": note
                },
                tags=[opts.dataset]
            )


        setup_seed(seed + i)
        opts.iteration = i + 1

        opts.resume = ''
        opts.start_epoch = 1
        train(opts, image_features, tokenizer, logger)

        opts.resume = 'SPD_best'
        val_metrics, test_metrics, epoch = test(opts, image_features, tokenizer, opts.wandb, logger)
        val_tcs_from_spd.append(val_metrics['TC'])
        test_tcs_from_spd.append(test_metrics['TC'])
        best_epochs_from_spd.append(epoch)

        print('')
        print('from SPD')
        print('val_tcs', val_tcs_from_spd)
        print('test_tcs', test_tcs_from_spd)
        print('VAL avg', round(np.mean(val_tcs_from_spd), 1), round(np.std(val_tcs_from_spd), 2))
        print('TEST avg', round(np.mean(test_tcs_from_spd), 1), round(np.std(test_tcs_from_spd), 2))
        print('')

        wandb.finish()

    # evaluations
    opts.resume = 'SPD_best'

    # uncomment if you want to run oracle studies after training
    # print('RUN ORACLE STUDIES')
    # print('ORIENTATION task only\n')
    # opts.config['oracle_initial_rotation'] = False
    # opts.config['oracle_directions'] = True
    # opts.config['oracle_stopping'] = True
    # main_test(opts, image_features, tokenizer)
    # print('\n\n')
    # print('DIRECTIONS task only\n')
    # opts.config['oracle_initial_rotation'] = True
    # opts.config['oracle_directions'] = False
    # opts.config['oracle_stopping'] = True
    # main_test(opts, image_features, tokenizer)
    # print('\n\n')
    # print('STOPPING task only')
    # opts.config['oracle_initial_rotation'] = True
    # opts.config['oracle_directions'] = True
    # opts.config['oracle_stopping'] = False
    # main_test(opts, image_features, tokenizer)

    print('\n\n\n\n')
    print('MAIN EVALUATION')
    #opts.DTW = True  # calculating DTW is very slow. Uncomment if needed
    opts.config['oracle_initial_rotation'] = False
    opts.config['oracle_directions'] = False
    opts.config['oracle_stopping'] = False
    main_test(opts, image_features, tokenizer)



def main_test(opts, image_features, tokenizer):
    all_val_metrics = dict(TC=list(), SPD=list(), SED=list(), DTW=list(), nDTW=list(), SDTW=list(), epoch=list())
    all_test_metrics = dict(TC=list(), SPD=list(), SED=list(), DTW=list(), nDTW=list(), SDTW=list())

    resume_orig = opts.resume
    opts.start_epoch = 1

    iterations = opts.config.get('iterations', 1)
    for i in range(iterations):
        opts.iteration = i + 1

        # set start time
        current_time = time.time() + 16 * 3600
        current_time = time.strftime('%Y-%m-%d__%H-%M-%S', time.localtime(current_time))
        opts.start_time = current_time
        if opts.log:
            # log
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            # 输出到文件
            file_handler = logging.FileHandler(os.path.join(opts.output_dir, 'log', f'{opts.start_time}_test.log'))
            logger.addHandler(file_handler)
            logger.info(f"interaction: {opts.iteration}")
            logger.info(opts)
        else:
            logger = None

        val_metrics, test_metrics, epoch = test(opts, image_features, tokenizer, False, logger)
        all_val_metrics['epoch'].append(epoch)
        for metric, value in val_metrics.items():
            all_val_metrics[metric].append(value)
        for metric, value in test_metrics.items():
            all_test_metrics[metric].append(value)

        opts.resume = resume_orig
    print('')
    print('')
    print('dev metrics')
    for key, values in all_val_metrics.items():
        if len(values) == 0 or key == 'DTW':
            continue
        print(key, ','.join(str(round(v, 1)) for v in values))
        print(key, '_mean', round(np.mean(values), 1))
        print('')
    print('')
    print('')
    print('test metrics')
    for key, values in all_test_metrics.items():
        if len(values) == 0 or key == 'DTW':
            continue
        print(key, ','.join(str(round(v, 1)) for v in values))
        print(key, '_mean', round(np.mean(values), 1))
        print('')
    print('')



def test(opts, image_features, tokenizer, ifwandb, logger):

    trainer = _load_trainer(opts, None, image_features, num_words=len(tokenizer))
    epoch = opts.start_epoch - 1

    assert opts.resume, 'The model was not resumed.'
    if opts.vlz_info_save:
        vlz_info_save = opts.resume.split('.')[0]
    else:
        vlz_info_save = None
    test_env = OutdoorVlnBatch(opts, image_features, batch_size=opts.batch_size, splits=['test'], tokenizer=tokenizer, name='test', vlz_info_save = vlz_info_save)
    val_env = OutdoorVlnBatch(opts, image_features, batch_size=opts.batch_size, splits=['dev'], tokenizer=tokenizer, name='eval', vlz_info_save=vlz_info_save)
    val_metrics = trainer.eval_(epoch, val_env, ifwandb, logger)
    test_metrics = trainer.eval_(epoch, test_env, ifwandb, logger)
    return val_metrics, test_metrics, epoch


def train(opts, image_features, tokenizer, logger=None):
    log_dir = os.path.join(opts.output_dir, 'tensorboard', 'iter{}'.format(opts.iteration))
    # tb_logger = set_tb_logger(log_dir, opts.resume)
    best_SPD, best_TC = float("inf"), 0.0

    train_env = OutdoorVlnBatch(opts, image_features, batch_size=opts.batch_size, splits=['train'], tokenizer=tokenizer, name="train", sample_bpe=opts.config.do_sample_bpe)
    trainer = _load_trainer(opts, train_env, image_features, num_words=len(tokenizer))

    val_seen_env = OutdoorVlnBatch(opts, image_features, batch_size=opts.batch_size, splits=['dev'], tokenizer=tokenizer, name="eval")

    for epoch in range(opts.start_epoch, opts.config.max_num_epochs + 1):
        trainer.train(epoch, train_env, opts.wandb, logger)
        if epoch % opts.eval_every_epochs == 0:

            val_metrics = trainer.eval_(epoch, val_seen_env, opts.wandb, logger=logger)
            TC = val_metrics['TC']
            SPD = val_metrics['SPD']
            is_best_SPD = SPD <= best_SPD
            is_best_TC = TC >= best_TC
            best_SPD = min(SPD, best_SPD)
            best_TC = max(TC, best_TC)
            print("--> Best dev TC: {}, best dev SPD: {}".format(best_TC, best_SPD))
            if logger is not None:
                logger.info("--> Best dev TC: {}, best dev SPD: {}".format(best_TC, best_SPD))

            ckpt = ({
                'SPD': SPD,
                'TC': TC,
                'opts': opts,
                'epoch': epoch + 1,
                'model_state_dict': trainer.agent.model.state_dict(),
                'instr_encoder_state_dict': trainer.agent.instr_encoder.state_dict()
            })
            save_checkpoint(ckpt, is_best_SPD, is_best_TC, epoch=epoch)

    print("--> Finished training")
    if logger is not None:
        logger.info("--> Finished training")


def _load_trainer(opts, train_env, image_features, num_words):
    # Build model, optimizers, agent and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    instr_encoder = InstructionEncoder(opts=opts,
                                       vocab_size=num_words,
                                       embedding_dim=opts.config.embedding_dim,
                                       dropout=opts.config.dropout).to(device)

    model = ORAR(opts, instr_encoder, image_features).to(device)

    if opts.config.optimizer == 'adamw':
        params = list(instr_encoder.named_parameters()) + list(model.named_parameters())
        print('use AdamW optimizer')
        decay = []
        no_decay = []
        for name, param in params:
            if not param.requires_grad:
                continue
            # print(param.shape, name)
            if len(param.shape) == 1 or 'layer_norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': opts.config.weight_decay}]
        optimizer = torch.optim.AdamW(params, lr=opts.config.learning_rate, weight_decay=0.)
    else:
        print('use Adam optimizer')
        params = list(instr_encoder.parameters()) + list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=opts.config.learning_rate, weight_decay=opts.config.weight_decay)

    if opts.resume:
        model, instr_encoder, best_SPD, best_TC = resume_training(opts, model, instr_encoder)

    agent = OutdoorVlnAgent(opts, train_env, instr_encoder, model)
    trainer = OutdoorVlnTrainer(opts, agent, optimizer)

    return trainer


if __name__ == "__main__":
    main(opts)
