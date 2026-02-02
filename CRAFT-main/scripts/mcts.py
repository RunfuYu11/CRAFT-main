import logging
import os
import re

import numpy as np
import pandas as pd
import json
import pickle
import datetime
import hydra
from omegaconf import DictConfig
import torch
import time
import warnings

import configs.config

warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')

from scripts.preprocess import make_counter, make_transforms
from Model.GCN import network
from Model.GCN.utils import template_prediction, check_templates

from scripts.beam_search import beam_decode, greedy_translate

from chemreact.utils.config import load_config as load_chem_cfg
from chemreact.data.tokenization import load_vocab as load_smiles_vocab
from chemreact.models.chem_g2s import ChemG2S

from Utils.utils import read_smilesset, RootNode, NormalNode, smi_tokenizer, MW_checker, is_empty
from Utils.reward import getReward


class MCTS():
    def __init__(self, init_smiles, model, GCN_model, vocab, Reward, max_depth=10, c=1, step=0, n_valid=0,
                 n_invalid=0, max_r=-1000, r_dict=None, src_transforms=None, beam_width=10, nbest=5,
                 inf_max_len=256, beam_templates: list = None, rollout_depth=None, device=None, GCN_device=None,
                 exp_num_sampling=None, roll_num_sampling=None):
        self.init_smiles = init_smiles
        self.model = model
        self.GCN_model = GCN_model
        self.vocab = vocab
        self.Reward = Reward
        self.max_depth = max_depth
        self.valid_smiles = {}
        self.terminate_smiles = {}
        self.c = c
        self.count = 0
        self.max_score = max_r
        self.step = step
        self.n_valid = n_valid
        self.n_invalid = n_invalid
        self.total_nodes = 0
        self.expand_max = 0
        self.r_dict = r_dict
        self.transforms = src_transforms
        self.beam_width = beam_width
        self.nbest = nbest
        self.inf_max_len = inf_max_len
        self.beam_templates = beam_templates
        self.rollout_depth = rollout_depth
        self.device = device
        self.GCN_device = GCN_device
        self.gen_templates = []
        self.num_sampling = exp_num_sampling
        self.roll_num_sampling = roll_num_sampling
        self.no_template = False
        self.smi_to_template = {}
        self.accum_time = 0

    def select(self):
        self.current_node = self.root
        while len(self.current_node.children) != 0:
            self.current_node = self.current_node.select_children()
            if self.current_node.depth + 1 > self.max_depth:
                tmp = self.current_node
                while self.current_node is not None:
                    self.current_node.cum_score += -1
                    self.current_node.visit += 1
                    self.current_node = self.current_node.parent
                tmp.remove_Node()
                self.current_node = self.root

    def expand(self):
        self.next_smiles = {}
        self.smi_to_template = {}
        self.expand_max = 0

        matched_indices = []
        input_smi = self.current_node.smi
        self.no_template = False
        indices = template_prediction(GCN_model=self.GCN_model, input_smi=input_smi,
                                      num_sampling=self.num_sampling, GCN_device=self.GCN_device)
        matched_indices = check_templates(indices, input_smi, self.r_dict)
        if len(matched_indices) != 0:
            self.gen_templates.extend(matched_indices)
            with torch.no_grad():
                for i in matched_indices:
                    input_conditional = smi_tokenizer(i + input_smi).split(' ')
                    input_tokens = self.transforms(input_conditional).to(self.device)
                    outputs = beam_decode(
                        v=self.vocab, model=self.model, input_tokens=input_tokens, template_idx=i,
                        device=self.device, inf_max_len=self.inf_max_len, beam_width=self.beam_width,
                        nbest=self.nbest, Temp=0.8, beam_templates=self.beam_templates
                    )
                    parse_fail = sum(1 for s in outputs if Chem.MolFromSmiles(s) is None)
                    mw_fail = sum(1 for s in outputs if
                                  (Chem.MolFromSmiles(s) is not None and not MW_checker(Chem.MolFromSmiles(s), 600)))

                    for output in outputs:
                        self.next_smiles[output] = 0
                        self.smi_to_template[output] = i
            self.check()
        else:
            self.no_template = True
            while (len(self.current_node.children) == 0) or (
                    min([cn.visit for cn in self.current_node.children]) >= 10000):
                self.current_node.cum_score = -10000
                self.current_node.visit = 10000
                self.current_node = self.current_node.parent

    def check(self):
        valid_list = []
        invalid_list = []
        score_que = []
        tmp = self.current_node

        if len(self.next_smiles) == 0:
            self.current_node.cum_score = -100000
            self.current_node.visit = 100000
            self.current_node.remove_Node()
            logging.info('0 molecules are expanded.')
        else:
            reaction_path = []
            while self.current_node.depth >= 0:
                reaction_path.insert(0, f'{self.current_node.template}.{self.current_node.smi}')
                self.current_node = self.current_node.parent
            self.current_node = tmp

            for smi in self.next_smiles.keys():
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    self.n_invalid += 1
                    invalid_list.append(smi)
                elif (mol is not None) and (MW_checker(mol, 600) == True):
                    score_que.append(smi)
                    self.n_valid += 1
                else:
                    invalid_list.append(smi)

            scores, _, _ = self.Reward.reward(score_que)
            if len(scores) != 0:
                valid_scores = []
                for smi, score in scores:
                    template = self.smi_to_template[smi]
                    path = reaction_path.copy()
                    path.append(f'{template}.{smi}')
                    path = '.'.join(path)
                    if score is not None:
                        self.valid_smiles[self.step, smi, path] = score
                        valid_list.append((score, smi))
                        valid_scores.append(score)
                        self.max_score = max(self.max_score, score)
                        self.expand_max = max(self.expand_max, score)
                for smi in invalid_list:
                    self.next_smiles.pop(smi, None)
                logging.info(f'{len(self.next_smiles)} molecules are expanded.')
            else:
                self.no_template = True
                while (len(self.current_node.children) == 0) or (
                        min([cn.visit for cn in self.current_node.children]) >= 100000):
                    self.current_node.cum_score = -100000
                    self.current_node.visit = 100000
                    self.current_node = self.current_node.parent

    def simulate(self):
        self.rollout_result = {}
        for orig_smi in self.next_smiles:
            depth = 0
            smi_que = [orig_smi]
            max_smi = None
            max_score = -10000
            while depth < self.rollout_depth:
                input_conditional = []
                for next_smi in smi_que:
                    if Chem.MolFromSmiles(next_smi) is not None:
                        indices = template_prediction(self.GCN_model, next_smi, num_sampling=self.roll_num_sampling,
                                                      GCN_device=self.GCN_device)
                        matched_indices = check_templates(indices, next_smi, self.r_dict)
                        for t in matched_indices:
                            input_conditional.append(smi_tokenizer(t + next_smi).split(' '))
                if is_empty(input_conditional) == False:
                    with torch.no_grad():
                        input_tokens = self.transforms(input_conditional).to(self.device)
                        output = greedy_translate(v=self.vocab, model=self.model, input_tokens=input_tokens,
                                                  inf_max_len=self.inf_max_len, device=self.device)
                    scores, max_smi_tmp, max_score_tmp = self.Reward.reward_remove_nan(output)
                    if max_score_tmp is None:
                        max_score_tmp = -10000
                    elif max_score < max_score_tmp:
                        max_score = max_score_tmp
                        max_smi = max_smi_tmp
                else:
                    break
                depth += 1
                smi_que = output
            if max_score > 0:
                self.next_smiles[orig_smi] = max_score
                self.rollout_result[orig_smi] = (max_smi, max_score)
            else:
                self.next_smiles[orig_smi] = 0

    def backprop(self):
        for key, value in self.next_smiles.items():
            child = NormalNode(smi=key, c=self.c)
            child.template = self.smi_to_template[key]
            child.cum_score += value
            child.imm_score = value
            child.id = self.total_nodes
            self.total_nodes += 1
            try:
                child.rollout_result = self.rollout_result[key]
            except KeyError:
                child.rollout_result = ('Termination', -10000)
            self.current_node.add_Node(child)
        max_reward = max(self.next_smiles.values())
        self.max_score = max(self.max_score, max_reward)
        while self.current_node is not None:
            self.current_node.visit += 1
            self.current_node.cum_score += max_reward
            self.current_node.imm_score = max(self.current_node.imm_score, max_reward)
            self.current_node = self.current_node.parent

    def search(self, n_step):
        n = NormalNode(self.init_smiles)
        self.root.add_Node(n)
        while self.step < n_step:
            self.step += 1
            if self.n_valid + self.n_invalid == 0:
                valid_rate = 0
            else:
                valid_rate = self.n_valid / (self.n_valid + self.n_invalid)
            logging.info(
                f'step:{self.step}, INIT_SCORE:{self.init_score}, MAX_SCORE:{self.max_score}, VALIDITY:{valid_rate}')
            self.select()
            logging.info(f'selected_score:{self.current_node.imm_score}')
            self.expand()
            expand_max = self.expand_max if self.expand_max != 0 else None
            if self.no_template == True:
                logging.info('no template')
                continue
            if len(self.next_smiles) != 0:
                self.simulate()
                self.backprop()


class ParseSelectMCTS(MCTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = RootNode()
        self.current_node = None
        self.next_smiles = {}
        self.rollout_result = {}
        scores, _, _ = self.Reward.reward([self.init_smiles])
        _, self.init_score = scores[0]


@hydra.main(config_path=None, config_name='config', version_base=None)
def main(cfg: DictConfig):
    date = datetime.datetime.now().strftime('%Y%m%d')
    num = 1
    while True:
        out_dir = hydra.utils.get_original_cwd() + f"{cfg['mcts']['out_dir']}/{date}_{num}"
        if os.path.isdir(out_dir):
            num += 1
            continue
        else:
            os.makedirs(out_dir, exist_ok=True)
            break
    logging.info(f'{out_dir} was created.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # baseline 预处理（保留）
    src_train_path = '/media/aita4090/YRF/CRAFT-main/data/USPTO/src_train.txt'
    tgt_train_path = '/media/aita4090/YRF/CRAFT-main/data/USPTO/tgt_train.txt'
    src_valid_path = '/media/aita4090/YRF/CRAFT-main/data/USPTO/src_valid.txt'
    tgt_valid_path = '/media/aita4090/YRF/CRAFT-main/data/USPTO/tgt_valid.txt'
    data_dict = make_counter(src_train_path=src_train_path,
                             tgt_train_path=tgt_train_path,
                             src_valid_path=src_valid_path,
                             tgt_valid_path=tgt_valid_path)
    src_transforms, _, v = make_transforms(data_dict=data_dict, make_vocab=True)

    # 输入起始分子
    init_smiles = read_smilesset('/media/aita4090/YRF/CRAFT-main/data/input/unseen_ZINC_AKT1.txt')
    n_valid = 0
    n_invalid = 0

    # 加载 ChemG2S（注意：需要 pad_id）
    chem_cfg = load_chem_cfg('/media/aita4090/YRF/CRAFT-main/configs/default.yaml')
    smiles_vocab = load_smiles_vocab(chem_cfg['data']['smiles_vocab'])  # dict[str->int]
    pad_id = smiles_vocab['<pad>']
    model = ChemG2S(chem_cfg, vocab_size=len(smiles_vocab), pad_id=pad_id).to(device)
    ckpt = torch.load('/media/aita4090/YRF/CRAFT-main/chemreact/train/checkpoints/best.pt', map_location=device)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    model.eval()

    # 挂载资源到模型
    with open('/media/aita4090/YRF/CRAFT-main/resources/template_id_vocab.json', 'r', encoding='utf-8') as f:
        tpl_id_vocab_raw = json.load(f)
    if "template_id_to_int" in tpl_id_vocab_raw:
        tpl_id_vocab = {str(k): int(v) for k, v in tpl_id_vocab_raw["template_id_to_int"].items()}
    else:
        tpl_id_vocab = {str(k): int(v) for k, v in tpl_id_vocab_raw.items()}
    with open('/media/aita4090/YRF/CRAFT-main/resources/template_roles.json', 'r', encoding='utf-8') as f:
        tpl_roles = json.load(f)
    with open('/media/aita4090/YRF/CRAFT-main/data/label_template.json', 'r', encoding='utf-8') as f:
        label_templates = json.load(f)
    with open('/media/aita4090/YRF/CRAFT-main/data/beamsearch_template_list.txt', 'r') as f:
        beam_templates = f.read().splitlines()

    model._chem_resources = {
        'smiles_vocab': smiles_vocab,
        'itos': {i: s for s, i in smiles_vocab.items()},
        'template_id_vocab': tpl_id_vocab,
        'template_roles': tpl_roles,
        'label_templates': label_templates,
        'beam_templates': set([re.sub(r'\D', '', x) for x in beam_templates]),
        'chem_cfg': chem_cfg
    }

    # 加载 GCN 模板预测
    dim_GCN = cfg['GCN_train']['dim']
    n_conv_hidden = cfg['GCN_train']['n_conv_hidden']
    n_mlp_hidden = cfg['GCN_train']['n_mlp_hidden']
    dropout = cfg['Model']['dropout']
    GCN_model = network.MolecularGCN(dim=dim_GCN, n_conv_hidden=n_conv_hidden,
                                     n_mlp_hidden=n_mlp_hidden, dropout=dropout).to(device)
    GCN_model.load_state_dict(torch.load('/media/aita4090/YRF/CRAFT-main/ckpts/GCN/GCN.pth', map_location=device))
    GCN_model.eval()

    # 回报函数
    reward = getReward(name=cfg['mcts']['reward_name'])
    logging.info('REWARD: %s', cfg['mcts']['reward_name'])
    with open('/media/aita4090/YRF/CRAFT-main/data/label_template.json') as f:
        r_dict = json.load(f)
    widths = [40]
    protein = cfg['mcts']['reward_name'] # 作为文件名前缀

    for width in widths:
        for start_smiles in init_smiles:
            start = time.time()
            mcts = ParseSelectMCTS(
                start_smiles, model=model, GCN_model=GCN_model, vocab=v, Reward=reward,
                max_depth=cfg['mcts']['max_depth'], step=0, n_valid=n_valid, n_invalid=n_invalid,
                c=cfg['mcts']['ucb_c'], max_r=reward.max_r, r_dict=r_dict, src_transforms=src_transforms,
                beam_width=width, nbest=width,
                beam_templates=beam_templates, rollout_depth=cfg['mcts']['rollout_depth'],
                roll_num_sampling=cfg['mcts']['roll_num_sampling'], device=device,
                GCN_device=device, exp_num_sampling=cfg['mcts']['exp_num_sampling']
            )
            mcts.search(n_step=cfg['mcts']['n_step'])
            reward.max_r = mcts.max_score
            n_valid += mcts.n_valid
            n_invalid += mcts.n_invalid
            end = time.time()
            logging.info('Elapsed Time (beam_width=%d): %f', width, (end - start))
            # 保存结果`
            out_dir = hydra.utils.get_original_cwd() + f"{cfg['mcts']['out_dir']}/{date}_{num}"
            generated_smiles = pd.DataFrame(columns=['SMILES', 'Reward', 'Imp', 'MW', 'step', 'reaction_path'])
            scores, _, _ = reward.reward([start_smiles])
            start_reward = scores[0][1]
            for kv in mcts.valid_smiles.items():
                step, smi, path = kv[0]
                step = int(step)
                try:
                    w = Descriptors.MolWt(Chem.MolFromSmiles(smi))
                except:
                    w = 0
                if (kv[1] is None) or (start_reward is None):
                    Imp = None
                else:
                    Imp = kv[1] - start_reward
                row = {'SMILES': smi, 'Reward': kv[1], 'Imp': Imp, 'MW': w, 'step': step, 'reaction_path': path}
                generated_smiles = pd.concat([generated_smiles, pd.DataFrame([row])], ignore_index=True)
            generated_smiles = generated_smiles.sort_values('Reward', ascending=False)

            # 文件名：蛋白名称_起始smiles_宽度.csv（对 SMILES 做文件名安全化）
            safe_smi = re.sub(r'[<>:"/\\|?*\s]+', '_', start_smiles)
            fname = f"{protein}_{safe_smi}_{width}.csv"
            generated_smiles.to_csv(os.path.join(out_dir, fname), index=False)


if __name__ == '__main__':
    main()
