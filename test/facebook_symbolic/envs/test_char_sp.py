from collections import namedtuple
import pytest
from src.envs import char_sp
import numpy as np


@pytest.mark.parametrize("task,data_path", [('prim_fwd',None)])
def test__init__(task,data_path, params,env):
    res = env.create_train_iterator(task,params,data_path)
    #a = next(iter(res))

# @pytest.mark.parametrize("nb_total_ops, max_int", [(8,1000)])
# def test__generate_expr(nb_total_ops,max_int):
#     rd = np.random.RandomState(0)

@pytest.fixture
def EnvDataset(params,env):
    task = "prim_fwd"
    data_path = None
    res = char_sp.EnvDataset(env,task,params=params,train=True,path=data_path,rng=np.random)
    return res

# def test_get_item_EnvDataset(EnvDataset):
#     a = next(iter(EnvDataset))

def test_gen_prim_fwd(EnvDataset):
    EnvDataset.env.gen_prim_fwd(EnvDataset.rng)
