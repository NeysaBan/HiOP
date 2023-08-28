import yaml
import math
import torch
import pytest
from rectime import show_time
from torch.utils.cpp_extension import load
import torch.nn.functional as F


with open("case/pytest/pyconfig.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
device = 'cuda:' + str(config['device_id'])

cuda_module = load(name="reduce_arr",
                        extra_include_paths=["include"],
                        sources=["reg_torch/reduce_op.cpp", "kernel/reduce_kernel.cu"],
                        build_directory="./build/",
                        verbose=True)

## test data
N = eval(config['N'])
smCnt = config['smCnt']
warpSize = config['warpSize']
blockSize = config['blockSize']
threadPerMultip = config['threadPerMultip']
dataPerBlock = 2 * blockSize ## 每个block可以处理的数据

input = torch.ones(size=(N, ), dtype=torch.float32).to(device)
## 填充input为block中所有线程都能工作起来的长度,会更方便处理input长度不能被2 x blockSize整除的长度
newLen = int((N + dataPerBlock - 1) / dataPerBlock) * dataPerBlock
# padLen = newLen - N
# N = newLen
# input = F.pad(input, pad=(0, padLen), value=0)
doutputBlock = torch.zeros_like(input).to(device)
doutputWarp = torch.zeros_like(input).to(device)
ground_truth = torch.sum(input)


gridSize = min(int((N + blockSize - 1) / blockSize), 
                int(smCnt * threadPerMultip / blockSize * warpSize))
workBlock = math.ceil(gridSize / 2)
print("gridSize: ", gridSize)

def torch_go():
    output = torch.sum(input)
    return output
    
def cuda_block_go():
    cuda_module.reduce_forward(doutputBlock, input, N, gridSize, True)
    return doutputBlock

def cuda_warp_go():
    cuda_module.reduce_forward(doutputWarp, input, N, gridSize, False)
    return doutputWarp

class Test_reduce:
    def test_torch(self):
        torch_res = show_time(torch_go, cuda=False, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, torch_res)
    # def test_cuda_block(self):
    #     cuda_res = show_time(cuda_block_go, cuda=True, device=device, ntest=config['ntest'])
    #     print(cuda_res)
    #     cuda_res = torch.sum(cuda_res[:gridSize])
    #     assert torch.allclose(ground_truth, cuda_res)
    def test_cuda_warp(self):
        cuda_res = show_time(cuda_warp_go, cuda=True, device=device, ntest=config['ntest'])
        cuda_res = torch.sum(cuda_res[:workBlock]) * 2
        assert torch.allclose(ground_truth, cuda_res)
    
if __name__ == '__main__':
    pytest.main(['-v', '-s', "case/pytest/reduce_test.py"])
    # pytest.main(['-v', '-s'])