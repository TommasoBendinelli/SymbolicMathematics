from collections import namedtuple
import pytest
d = {'dump_path': './dumped/first_exp/5lxzpmvyke', 'exp_name': 'first_exp', 'save_periodic': 0, 
'exp_id': '5lxzpmvyke', 'fp16': False, 'amp': -1, 'emb_dim': 256, 'n_enc_layers': 4, 'n_dec_layers': 4,
'n_heads': 4, 'dropout': 0, 'attention_dropout': 0, 'share_inout_emb': True, 'sinusoidal_embeddings': False,
'env_base_seed': 0, 'max_len': 512, 'batch_size': 32, 'optimizer': 'adam,lr=0.0001', 'clip_grad_norm': 5,
'epoch_size': 300000, 'max_epoch': 100000, 'stopping_criterion': '', 'validation_metrics': '',
'accumulate_gradients': 1, 'num_workers': 10, 'same_nb_ops_per_batch': False, 'export_data': True, 
'reload_data': '', 'reload_size': -1, 'env_name': 'char_sp', 
'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1', 
'max_ops': 10, 'max_ops_G': 4, 'max_int': 10000, 'int_base': 10, 'balanced': False, 'precision': 10,
'positive': False, 'rewrite_functions': '', 'leaf_probs': '0.75,0,0.25,0', 'n_variables': 1, 'n_coefficients': 0,
'clean_prefix_expr': True, 'tasks': ['prim_fwd'], 'beam_eval': False, 'beam_size': 1, 'beam_length_penalty': 1,
'beam_early_stopping': True, 'reload_model': '', 'reload_checkpoint': '', 'eval_only': False, 'eval_verbose': 0,
'eval_verbose_print': False, 'debug_slurm': True, 'debug': False, 'cpu': True, 'local_rank': 0, 'master_port': -1, 
'is_slurm_job': False, 'n_nodes': 1, 'node_id': 0, 'global_rank': 0, 'world_size': 1, 'n_gpu_per_node': 1,
'is_master': True, 'multi_node': False, 'multi_gpu': False,
'command': 'python /Users/tommaso/repos/SymbolicMathematics/main.py --export_data true --cpu True\
--tasks prim_fwd --exp_name first_exp --operators \'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1\'\
--exp_id "5lxzpmvyke"', 'n_words': 91, 'eos_index': 0, 'pad_index': 1, 'export_path_prefix': './dumped/first_exp/5lxzpmvyke/data.prefix', 'export_path_infix': './dumped/first_exp/5lxzpmvyke/data.infix'}

ParamClass = namedtuple("params", d)
params = ParamClass(**d)


@pytest.mark.parametrize("task, params,data_path", [('prim_fwd',params,None)])
def test__init__(task,params,data_path):
    pass