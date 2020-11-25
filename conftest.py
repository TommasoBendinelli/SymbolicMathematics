from src.envs import ENVS, build_env
import pytest
from collections import namedtuple
from dataclasses import dataclass

@pytest.fixture(scope="session")
def params():
    @dataclass
    class Params:
        dump_path: str
        exp_name: str 
        save_periodic: int
        exp_id: str
        fp16: bool
        amp: int
        emb_dim: int
        n_enc_layers: int
        n_dec_layers: int
        n_heads: int
        dropout: int
        attention_dropout: int
        share_inout_emb: bool
        sinusoidal_embeddings: bool
        env_base_seed: int
        max_len: int
        batch_size: int
        optimizer: str
        clip_grad_norm: int
        epoch_size: int
        max_epoch: int
        stopping_criterion: str
        validation_metrics: str
        accumulate_gradients: int
        num_workers: int
        same_nb_ops_per_batch: str
        export_data: str
        validation_metrics: str
        reload_data: str
        reload_size: int
        env_name: str
        operators: str
        max_ops: int
        max_ops_G: int
        max_int: int
        int_base: int
        balanced: bool
        precision: int
        positive: bool
        rewrite_functions: str
        leaf_probs: str
        n_variables: int
        n_coefficients: int
        clean_prefix_expr: bool
        tasks: str
        beam_eval: bool
        beam_size: int
        beam_length_penalty: int
        beam_early_stopping: bool
        reload_model: str
        reload_checkpoint: str
        eval_only: bool
        eval_verbose: int
        eval_verbose_print: bool
        debug_slurm: bool
        debug: bool
        cpu: bool
        local_rank: int
        master_port: int
        is_slurm_job: bool
        n_nodes: int
        node_id: int
        global_rank: int
        world_size: int
        n_gpu_per_node: int
        is_slurm_job: bool
        n_nodes: int
        node_id: int
        global_rank: int
        world_size: int
        n_gpu_per_node: int
        is_master: bool
        multi_node: bool
        multi_gpu: bool
        command: str
        n_words: int
        eos_index: int
        pad_index: int
        export_path_prefix: str
        export_path_infix: str

        



    d = {'dump_path': './dumped/first_exp/5lxzpmvyke', 'exp_name': 'first_exp', 'save_periodic': 0, 
        'exp_id': '5lxzpmvyke', 'fp16': False, 'amp': -1, 'emb_dim': 256, 'n_enc_layers': 4, 'n_dec_layers': 4,
        'n_heads': 4, 'dropout': 0, 'attention_dropout': 0, 'share_inout_emb': True, 'sinusoidal_embeddings': False,
        'env_base_seed': 0, 'max_len': 512, 'batch_size': 32, 'optimizer': 'adam,lr=0.0001', 'clip_grad_norm': 5,
        'epoch_size': 300000, 'max_epoch': 100000, 'stopping_criterion': '', 'validation_metrics': '',
        'accumulate_gradients': 1, 'num_workers': 0, 'same_nb_ops_per_batch': False, 'export_data': True, 
        'reload_data': '', 'reload_size': -1, 'env_name': 'char_sp', 
        'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1', 
        'max_ops': 10, 'max_ops_G': 4, 'max_int': 5, 'int_base': 10, 'balanced': False, 'precision': 10,
        'positive': False, 'rewrite_functions': '', 'leaf_probs': '0.75,0,0.25,0', 'n_variables': 3, 'n_coefficients': 0,
        'clean_prefix_expr': True, 'tasks': 'prim_fwd', 'beam_eval': False, 'beam_size': 1, 'beam_length_penalty': 1,
        'beam_early_stopping': True, 'reload_model': '', 'reload_checkpoint': '', 'eval_only': False, 'eval_verbose': 0,
        'eval_verbose_print': False, 'debug_slurm': True, 'debug': False, 'cpu': True, 'local_rank': 0, 'master_port': -1, 
        'is_slurm_job': False, 'n_nodes': 1, 'node_id': 0, 'global_rank': 0, 'world_size': 1, 'n_gpu_per_node': 1,
        'is_master': True, 'multi_node': False, 'multi_gpu': False,
        'command': 'python /Users/tommaso/repos/SymbolicMathematics/main.py --export_data true --cpu True\
        --tasks prim_fwd --exp_name first_exp --operators \'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1\'\
        --exp_id "5lxzpmvyke"', 'n_words': 91, 'eos_index': 0, 'pad_index': 1, 'export_path_prefix': './dumped/first_exp/5lxzpmvyke/data.prefix', 'export_path_infix': './dumped/first_exp/5lxzpmvyke/data.infix'}
    #ParamClass = namedtuple("params", d)
    params = Params(**d)
    return params

@pytest.fixture(scope="session")
def env(params):
    env = build_env(params)
    return env