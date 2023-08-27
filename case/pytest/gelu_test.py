import yaml
import torch
import pytest
import ctypes
from rectime import show_time
from torch.utils.cpp_extension import load
import torch.nn.functional as F

with open("case/pytest/pyconfig.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
device = 'cuda:' + str(config['device_id'])


extra_cflags = ['-std=c++14']


cuda_module = load(name="gelu",
                        extra_include_paths=["include"],
                        sources=["reg_torch/gelu_op.cpp", "kernel/gelu_kernel.cu"],
                        build_directory="./build/",
                        verbose=True)

## test data
N = 1024 * 1024
blockSize = 256
dataPerBlock = blockSize

src = torch.randint(low = -10, high = 10, size=(N, ), dtype=torch.float32).half().to(device)
ddst = torch.zeros_like(src, dtype=torch.float32).half().to(device)
ground_truth = F.gelu(src)

gridSize = int((N + blockSize - 1) / blockSize)
print("gridSize: ", gridSize)

def torch_go():
    output = F.gelu(src)
    return output

def cuda_go():
    cuda_module.gelu_forward(ddst, src, N, gridSize)
    return ddst

class Test_gelu:
    def test_torch(self):
        torch_res = show_time(torch_go, cuda=False, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, torch_res)
    def test_cuda(self):
        cuda_res = show_time(cuda_go, cuda=True, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, cuda_res, atol=0.001)

if __name__ == '__main__':
    pytest.main(['-v', '-s', "case/pytest/gelu_test.py"])