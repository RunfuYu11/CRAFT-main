import math

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 128
    label_smoothing: float = 0.0
    lr: float = 0.001
    betas: tuple = (0.9, 0.998)
    step_num: int = 500000  # set training steps
    patience: int = 10
    log_interval: int = 100
    val_interval: int = 1000
    save_interval: int = 10000

@dataclass
class MCTSConfig:
    n_step: int = 100
    max_depth: int = 5
    out_dir: str = '/mcts_out'
    ucb_c: float = 1 / math.sqrt(2)
    reward_name: str = 'AKT1'  # 'DRD2' or 'AKT1' or 'CXCR4'
    beam_width: int = 10
    nbest: int = 10
    exp_num_sampling: int = 10
    rollout_depth: int = 2
    roll_num_sampling: int = 5

    chem_cfg_yaml: str = '/media/aita4090/YRF/CRAFT-main/configs/default.yaml'
    chem_ckpt: str = '/media/aita4090/YRF/CRAFT-main/chemreact/train/checkpoints/best.pt'
    chem_smiles_vocab: str = '/media/aita4090/YRF/CRAFT-main/resources/smiles_vocab.json'
    chem_template_id_vocab: str = '/media/aita4090/YRF/CRAFT-main/resources/template_id_vocab.json'
    chem_template_roles: str = '/media/aita4090/YRF/CRAFT-main/resources/template_roles.json'
    chem_label_template: str = '/media/aita4090/YRF/CRAFT-main/data/label_template.json'


@dataclass
class Config:
    train: TrainConfig = TrainConfig()
    mcts: MCTSConfig = MCTSConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)