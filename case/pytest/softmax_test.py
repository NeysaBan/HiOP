import yaml
import math
import torch
import pytest
from rectime import show_time
from torch.utils.cpp_extension import load

with open("case/pytest/pyconfig.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
device = 'cuda:' + str(config['device_id'])

cuda_module = load(name="softmax",
                        extra_include_paths=["include"],
                        sources=["reg_torch/softmax_op.cpp", "kernel/softmax_kernel.cu"],
                        build_directory="./build/",
                        verbose=True)


N = eval(config['N'])
smCnt = config['smCnt']
warpSize = config['warpSize']
blockSize = config['blockSize']
# threadPerMultip = config['threadPerMultip']

input = torch.ones(size=(N, ), dtype=torch.float32).to(device)
doutput = torch.zeros_like(input).to(device)
ground_truth = torch.softmax(input, dim=0).to(device)

row = int(math.sqrt(N))
assert row * row == N, "sqrt(N) failed"
col = row

gridX = 1
gridY = 125
blockX = 32
blockY = 8
blockNum = gridX * gridY
threadNum = blockX * blockY

def torch_go():
    output = torch.softmax(input, dim=0).to(device)
    return output

def cuda_go():
    cuda_module.softmax_forward(doutput, input, row, col, 
                                blockNum)
    return doutput

class Test_softmax:
    def test_torch(self):
        torch_res = show_time(torch_go, cuda=False, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, torch_res)
    def test_cuda_warp(self):
        cuda_res = show_time(cuda_go, cuda=True, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, cuda_res)

if __name__ == '__main__':
    pytest.main(['-v', '-s', "case/pytest/softmax_test.py"])
    # pytest.main(['-v', '-s'])