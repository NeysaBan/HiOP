import yaml
import torch
import torch.nn as nn
import pytest
import numpy as np
from rectime import show_time
from torch.autograd import Function
from torch.utils.cpp_extension import load


with open("case/pytest/pyconfig.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
device = 'cuda:' + str(config['device_id'])

cuda_module = load(name="reduce_arr",
                        extra_include_paths=["include"],
                        sources=["reg_torch/reduce_op.cpp", "kernel/reduce_kernel.cu"],
                        verbose=True)

## test data
N = 100000
input = torch.rand(size=(1, N), dtype=torch.float32).to(device)
output = torch.empty(1).to(device)
doutput = torch.empty(1).to(device)

def torch_go():
    output = torch.sum(input)
    
def cuda_go():
    cuda_module.torch_launch_reduce(doutput, input, N)


class Test_reduce:
    def test_torch(self):
        show_time(torch_go, cuda=False, device=device, ntest=config['ntest'])
    def test_cuda(self):
        show_time(cuda_go, cuda=True, device=device, ntest=config['ntest'])
        assert torch.allclose(output, doutput)
    
if __name__ == '__main__':
    pytest.main(['-v', '-s'])