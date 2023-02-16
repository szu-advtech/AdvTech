# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from Projector import Projector
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

from env import R2RBatch
import utils
from utils import padding_idx, print_progress
import model_OSCAR, model_PREVALENT
import param
from param import args
from collections import defaultdict


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': ([0],[-1], [0]), # left
      'right': ([0], [1], [0]), # right
      'up': ([0], [0], [1]), # up
      'down': ([0], [0],[-1]), # down
      'forward': ([1], [0], [0]), # forward
      '<end>': ([0], [0], [0]), # <end>
      '<start>': ([0], [0], [0]), # <start>
      '<ignore>': ([0], [0], [0])  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        ######################
        self.Projector = Projector().cuda()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda()
        ######################

        # Models
        if args.vlnbert == 'oscar':
            self.vln_bert = model_OSCAR.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_OSCAR.Critic().cuda()
        elif args.vlnbert == 'prevalent':
            self.vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_PREVALENT.Critic().cuda()
        self.models = (self.vln_bert, self.critic)

        # Optimizers

        #############
        self.Projector_optimizer = args.optimizer(self.Projector.parameters(), lr=args.lr)
        self.gpt2_optimizer = args.optimizer(self.gpt2.parameters(), lr=args.lr)
        self.optimizers = (self.Projector_optimizer,self.gpt2_optimizer)
        #############
        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.ndtw_criterion = utils.ndtw_initialize()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs):
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != padding_idx)

        token_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])

        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

                state = self.env.env.sims[idx].getState()[0]
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Language input        # 这里是个排序，把句子长度从长到短排序，perm_idx是它们原本的顺序
        sentence, language_attention_mask, token_type_ids, \
            seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx] #这是8个路径的信息

        #######################
        abc_index = " a b c d e f g h i j k l m n o p q r s t u v w x y z . ;"
        abc_tok = self.tokenizer(abc_index, return_tensors='pt')["input_ids"].cuda()
        abc_index = self.gpt2.transformer.wte(abc_tok)

        youcan = "\n\nThe actions you can choose are :"
        youcan = self.gpt2.transformer.wte(self.tokenizer(youcan, return_tensors='pt')["input_ids"].cuda())

        youchoose = "Choose a letter to indicate the direction of progress. The letter you choose is :"
        youchoose = self.gpt2.transformer.wte(self.tokenizer(youchoose, return_tensors='pt')["input_ids"].cuda())

        tasks = "You are a navigation robot." \
                " You need to go to the destination according to the instructions and then stop." \
                " Your instruction is :"
        self.tokenizer.add_special_tokens({'pad_token': '0'})
        tasks = [(tasks+ob['instructions']) for ob in perm_obs]
        tasks = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=tasks,
                                          padding=True,
                                          truncation=True,
                                          max_length=100,
                                          return_tensors='pt')
        tasks_attention_mask = tasks["attention_mask"].cuda()
        tasks_emb = self.gpt2.transformer.wte(tasks["input_ids"].cuda())

        input = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=tasks,
                                          padding=True,
                                          truncation=True,
                                          max_length=100,
                                          return_tensors='pt')
        input.data.pop("input_ids")

        tasks_emb_tmps_last = None
        tasks_attention_mask_tmps_last = None
        past = None

        ########################

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in perm_obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        ############
        self.episode_len = 10
        ############

        for t in range(self.episode_len): # 这里的episode是最大行动次数，在args里面设定的，是15

            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            ##########################################
            candidate_feat_projector = self.Projector(candidate_feat)# 把(8,动作数,2052)投影到(8,动作数,768)

            for v in range(len(candidate_leng)):# 遍历每个batchsize
                if(t == 0):# 第一次的时候把指令拼接在最前面
                    tasks_emb_tmp = torch.cat((tasks_emb[v, :, :],youcan[0,:,:]),0)
                    tasks_attention_mask_tmp = torch.cat((tasks_attention_mask[v, :],torch.ones(youcan.shape[1]).cuda()),0)
                else:
                    tasks_emb_tmp = youcan[0,:,:]
                    tasks_attention_mask_tmp = torch.ones(youcan.shape[1]).cuda()
                for n in range(max(candidate_leng)):
                    # tasks_emb[v, :, :] += abc_index[0, n, :] + abc_index[0 ,26,:]  \
                    #                 + candidate_feat_projector[v, n ,:] \
                    #                 + abc_index[0 ,27,:] # 分号
                    tasks_emb_tmp =  torch.cat((tasks_emb_tmp
                                                ,abc_index[0, n, :].unsqueeze(dim=0)
                                                ,abc_index[0 ,26,:].unsqueeze(dim=0)
                                                ,candidate_feat_projector[v, n ,:].unsqueeze(dim=0)
                                                ,abc_index[0 ,27,:].unsqueeze(dim=0)),0)# 分号
                    if(n<candidate_leng[v]):
                        tasks_attention_mask_tmp = torch.cat((tasks_attention_mask_tmp, torch.ones(4).cuda()), 0)

                    else:
                        tasks_attention_mask_tmp = torch.cat((tasks_attention_mask_tmp, torch.zeros(4).cuda()), 0)

                # tasks_emb[v, :, :] += youchoose
                tasks_emb_tmp = torch.cat((tasks_emb_tmp,youchoose[0,:,:]),0)
                tasks_attention_mask_tmp = torch.cat((tasks_attention_mask_tmp, torch.ones(youchoose.shape[1]).cuda()), 0)
                if(v ==0):
                    tasks_emb_tmps = tasks_emb_tmp.unsqueeze(dim=0)
                    tasks_attention_mask_tmps = tasks_attention_mask_tmp.unsqueeze(dim=0)
                else:

                    tasks_emb_tmps = torch.cat((tasks_emb_tmps,tasks_emb_tmp.unsqueeze(dim=0)),0)
                    tasks_attention_mask_tmps = torch.cat((tasks_attention_mask_tmps,tasks_attention_mask_tmp.unsqueeze(dim=0)),0)



            ##########################################

            ##############################

            # tasks.data.pop("input_ids")
            # tasks.data["inputs_embeds"] = tasks_emb_tmps
            # tasks.data["attention_mask"] = tasks_attention_mask_tmps
            # output = self.gpt2(**tasks.data)

            if(tasks_emb_tmps_last != None):
                tasks_emb_tmps = torch.cat((tasks_emb_tmps_last,tasks_emb_tmps),dim=1)
                tasks_attention_mask_tmps = torch.cat((tasks_attention_mask_tmps_last,tasks_attention_mask_tmps),dim=1)

            if (past != None):
                input.data["past_key_values"] = past

            input.data["inputs_embeds"] = tasks_emb_tmps
            input.data["attention_mask"] = tasks_attention_mask_tmps

            # tasks_emb_tmps_last = tasks_emb_tmps
            # tasks_attention_mask_tmps_last = tasks_attention_mask_tmps


            output=self.gpt2(**input)

            logit = torch.ones((batch_size,max(candidate_leng))).cuda() * float("-inf")
            for m in range(batch_size):
                for l in range(candidate_leng[m]):
                    logit[m,l] = output[0][m,-1,abc_tok[0,l]]
            ##############################

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
                ##################################################
                # 把正确答案填上去
                for i in range(batch_size):
                    if i == 0:
                        answer = abc_index[0, target[i], :].unsqueeze(dim=0).unsqueeze(dim=0)
                        answer_mask = torch.ones(1).cuda().unsqueeze(dim=0)
                    else:
                        answer = torch.cat((answer, abc_index[0, target[i], :].unsqueeze(dim=0).unsqueeze(dim=0)),dim=0)
                        answer_mask = torch.cat((answer_mask, torch.ones(1).cuda().unsqueeze(dim=0)),dim=0)

                # tasks_emb_tmps_last = torch.cat((tasks_emb_tmps,answer),dim=1)
                tasks_attention_mask_tmps_last = torch.cat((tasks_attention_mask_tmps,answer_mask),dim=1)
                tasks_emb_tmps_last = answer
                # tasks_attention_mask_tmps_last = answer_mask
                past = output["past_key_values"]
                ##################################################
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
                ##################################################
                # 把选择的答案填上去
                for i in range(batch_size):
                    if i == 0:
                        answer = abc_index[0, a_t[i], :].unsqueeze(dim=0).unsqueeze(dim=0)
                        answer_mask = torch.ones(1).cuda().unsqueeze(dim=0)
                    else:
                        answer = torch.cat((answer, abc_index[0, a_t[i], :].unsqueeze(dim=0).unsqueeze(dim=0)), dim=0)
                        answer_mask = torch.cat((answer_mask, torch.ones(1).cuda().unsqueeze(dim=0)), dim=0)

                # tasks_emb_tmps_last = torch.cat((tasks_emb_tmps,answer),dim=1)
                tasks_attention_mask_tmps_last = torch.cat((tasks_attention_mask_tmps, answer_mask), dim=1)
                tasks_emb_tmps_last = answer
                # tasks_attention_mask_tmps_last = answer_mask
                past = output["past_key_values"]
                ##################################################
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
                ##################################################
                # 把选择的答案填上去
                for i in range(batch_size):
                    if i == 0:
                        answer = abc_index[0, a_t[i], :].unsqueeze(dim=0).unsqueeze(dim=0)
                        answer_mask = torch.ones(1).cuda().unsqueeze(dim=0)
                    else:
                        answer = torch.cat((answer, abc_index[0, a_t[i], :].unsqueeze(dim=0).unsqueeze(dim=0)),dim=0)
                        answer_mask = torch.cat((answer_mask, torch.ones(1).cuda().unsqueeze(dim=0)),dim=0)

                # tasks_emb_tmps_last = torch.cat((tasks_emb_tmps,answer),dim=1)
                tasks_attention_mask_tmps_last = torch.cat((tasks_attention_mask_tmps,answer_mask),dim=1)
                tasks_emb_tmps_last = answer
                # tasks_attention_mask_tmps_last = answer_mask
                past = output["past_key_values"]
                ##################################################
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]            # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:  # 这部分被我去掉了内部奖励，只剩下外部奖励（我猜
            # Last action in A2C
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            '''
            language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
            '''
            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            # self.vln_bert.vln_bert.config.directions = max(candidate_leng)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward

            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()


                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                ''''''
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss

                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback


        ##############################
        self.Projector.train()
        feedback = 'sample'
        ##############################


        self.losses = []
        for iter in range(1, n_iters + 1):

            ##############################
            self.Projector_optimizer.zero_grad()
            self.gpt2_optimizer.zero_grad()
            ##############################

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)

            elif feedback == 'sample':  # agents in IL and RL separately

                # 这样就变成了student forcing
                self.feedback = 'sample'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)

            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

            #######################################
            self.Projector_optimizer.step()
            self.gpt2_optimizer.step()
            #######################################

            if args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)
            else:
                print_progress(iter, n_iters + 1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
