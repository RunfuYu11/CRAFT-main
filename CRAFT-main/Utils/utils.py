import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def smi_tokenizer(smi):
    import re
    pattern =  '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

class Node:
    def __init__(self):
        self.parent = None
        self.template = None
        self.path = []
        self.depth = -100
        self.visit = 1
        self.children = []
        self.imm_score = 0
        self.cum_score = 0
        self.c = 1
        self.id = -1
        self.rollout_result = ('None', -1000)

    def add_Node(self, c):
        c.parent = self
        c.depth = self.depth + 1
        self.children.append(c)

    def calc_UCB(self):
        if self.visit == 0:
            ucb = 1e+6
        else:
            ucb = self.cum_score/self.visit + self.c*math.sqrt(2*math.log(self.parent.visit)/self.visit)
        return ucb

    def select_children(self):
        children_ucb = []
        for cn in self.children:
            children_ucb.append(cn.calc_UCB())
        max_ind = np.random.choice(np.where(np.array(children_ucb) == max(children_ucb))[0])
        return self.children[max_ind]

    def select_children_rand(self):
        indices = list(range(0, len(self.children)))
        ind = np.random.choice(indices)
        return self.children[ind]


class RootNode(Node):
    def __init__(self, c=1/np.sqrt(2)):
        super().__init__()
        self.smi = '&&'
        self.depth = -1

        self.c = c

class NormalNode(Node):
    def __init__(self, smi, c=1/np.sqrt(2)):
        super().__init__()
        self.smi = smi
        self.c = c
        self.template = None

    def remove_Node(self):
        self.parent.children.remove(self)

def read_smilesset(path):
    smiles_list = []
    with open(path) as f:
        for smiles in f:
            smiles_list.append(smiles.rstrip())

    return smiles_list

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, batch_size, v=None):
    pad_mask = (target != v['<pad>']) 
    true_pos = torch.nonzero(pad_mask).squeeze().tolist()
    out_extracted = output[true_pos]
    t_extracted = target[true_pos]
    _, pred = out_extracted.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(t_extracted.reshape(1, -1).expand_as(pred))
    correct_rate = (correct[0].float().sum(0, keepdim=True)) / len(t_extracted)

    target = target.reshape(-1, batch_size)
    output = output.reshape(-1, batch_size, v.__len__())
    _, pred = output.topk(1, 2, True, True)
    pred = pred.squeeze()
    correct_cum = 0
    EOS_token = v['<eos>']
    for i in range(batch_size):
        t = target[:, i].tolist()
        eos_idx = t.index(EOS_token)
        t = t[0:eos_idx]
        p = pred[:, i].tolist()
        p = p[0:len(t)]
        if t == p:
            correct_cum += 1
    perfect_acc = correct_cum / batch_size
    return correct_rate.item(), perfect_acc

def calc_topk_perfect_acc(x, target, batch_size, EOS):
    correct_cum = 0
    if x.dim() < 3:
        x = x.unsqueeze(-1)
    for i in range(batch_size):
        t = target[:, i].tolist()
        eos_idx = t.index(EOS)
        t = t[0:eos_idx]
        for j in range(x.size(2)):
            p = x[:, i, j].tolist()
            p = p[0:len(t)]
            if t == p:
                correct_cum += 1
                break
    return correct_cum / batch_size
    

def MW_checker(mol, threshold:int = 750):
    mol.UpdatePropertyCache(strict=False)
    MW = Descriptors.ExactMolWt(mol)
    if MW > threshold:
        return False
    else:
        return True

def is_empty(li):
    return all(not sublist for sublist in li)

def torch_fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    