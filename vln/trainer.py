import pickle

import torch
import wandb

from utils import AverageMeter
import time
import math


class OutdoorVlnTrainer:
    def __init__(self, opts, agent, optimizer):
        self.opts = opts
        self.agent = agent
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, train_env, ifwandb, logger=None):
        print('Training on {} env ...'.format(train_env.splits[0]))
        print('Learning rate: {}'.format(self.optimizer.param_groups[0]['lr']))

        if logger is not None:
            logger.info('Training on {} env ...'.format(train_env.splits[0]))
            logger.info('Learning rate: {}'.format(self.optimizer.param_groups[0]['lr']))

        self.agent.env = train_env
        self.agent.model.train()
        self.agent.instr_encoder.train()
        self.agent.env.reset_epoch()

        losses = AverageMeter()
        action_loss = AverageMeter()
        subInstr_loss = AverageMeter()
        progress_loss = AverageMeter()

        batch_time = AverageMeter()

        end = time.time()
        self.train_iters_epoch = math.ceil(len(train_env.data) / self.opts.batch_size)

        for iter_ in range(1, self.train_iters_epoch + 1):
            loss, loss_list, _, _ = self.agent.rollout(is_test=False)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            losses.update(loss.item(), len(self.agent.env.batch))
            action_loss.update(loss_list['action_loss'], len(self.agent.env.batch))
            progress_loss.update(loss_list['progress_loss'], len(self.agent.env.batch))
            subInstr_loss.update(loss_list['subInstr_loss'], len(self.agent.env.batch))
            end = time.time()


            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                epoch, iter_, self.train_iters_epoch, batch_time=batch_time,
                loss=losses), end='')
            if logger is not None:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, iter_, self.train_iters_epoch, batch_time=batch_time,
                    loss=losses))

        if ifwandb:
            wandb.log({"loss": losses.avg, "action_loss": action_loss.avg, "progress_loss": progress_loss.avg, "subInstr_loss": subInstr_loss.avg, "learning_rate": self.optimizer.param_groups[0]['lr']}, step=epoch)


    def eval_(self, epoch, val_env, ifwandb, logger=None):
        phase = val_env.env.name
        print('Evaluating on {} env ...'.format(phase))
        if logger is not None:
            logger.info('Evaluating on {} env ...'.format(phase))

        losses = AverageMeter()
        batch_time = AverageMeter()

        self.agent.env = val_env
        self.agent.env.reset_epoch()
        self.agent.model.eval()
        self.agent.instr_encoder.eval()

        val_iters_epoch = math.ceil(len(val_env.data) / self.opts.batch_size)

        metrics = [0] * 3  # [TC, SPD, SED]
        if self.opts.CLS:
            metrics += [0]
        if self.opts.DTW:
            metrics += [0] * 5
        with torch.no_grad():
            end = time.time()
            vlz_info_all = []
            for iter_ in range(1, val_iters_epoch + 1):
                _, trajs, vlz_info = self.agent.rollout(is_test=True)
                #print_actions(agent_actions)
                self.agent.env.eva_metrics(trajs, metrics)
                batch_time.update(time.time() - end)
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    epoch, iter_, val_iters_epoch, batch_time=batch_time))
                if logger is not None:
                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        epoch, iter_, val_iters_epoch, batch_time=batch_time))

                vlz_info_all.append(vlz_info)
            if val_env.vlz_info_save is not None:
                pickle.dump(vlz_info_all, open(val_env.vlz_info_save, 'wb'))

        metrics = [m / len(val_env.data) for m in metrics]
        metrics = [m * 100 if m < 1 else m for m in metrics]
        if ifwandb:
            wandb.log({"TC": metrics[0], "SPD": metrics[1], "SED": metrics[2]}, step=epoch)

        d_metrics = dict(TC=metrics[0], SPD=metrics[1], SED=metrics[2])
        print("=======[%s] Evaluation Metrics=======" % phase)
        print("TC: %.2f, SPD: %.2f, SED: %.2f" % tuple(metrics[:3]), end='')
        if logger is not None:
            logger.info("=======[%s] Evaluation Metrics=======" % phase)
            logger.info("TC: %.2f, SPD: %.2f, SED: %.2f" % tuple(metrics[:3]))
        if self.opts.CLS:
            print(', CLS:%.2f' % metrics[3], end='')
            if logger is not None:
                logger.info(', CLS:%.2f' % metrics[3])
            d_metrics['CLS'] = metrics[3]
        if self.opts.DTW:
            print(', DTW:%.2f, nDTW:%.2f, SDTW:%.2f' % tuple(metrics[-3:]))
            if logger is not None:
                logger.info(', DTW:%.2f, nDTW:%.2f, SDTW:%.2f' % tuple(metrics[-3:]))
            d_metrics['DTW'] = metrics[-3]
            d_metrics['nDTW'] = metrics[-2]
            d_metrics['SDTW'] = metrics[-1]
        else:
            print('')
            if logger is not None:
                logger.info('')
        print("================================")
        if logger is not None:
            logger.info("================================")

        return d_metrics
