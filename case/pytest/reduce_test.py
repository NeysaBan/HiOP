import yaml
import torch
import pytest
from rectime import show_time
from torch.utils.cpp_extension import load


with open("case/pytest/pyconfig.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
device = 'cuda:' + str(config['device_id'])

cuda_module = load(name="reduce_arr",
                        extra_include_paths=["include"],
                        sources=["reg_torch/reduce_op.cpp", "kernel/reduce_kernel.cu"],
                        build_directory="./build/",
                        verbose=True)

## test data
N = 1024 * 1024 * 1024
input = torch.ones(size=(N, ), dtype=torch.float32).to(device)
doutput = torch.zeros_like(input).to(device)
ground_truth = torch.sum(input)

blockSize = 256
gridSize = int((N + blockSize - 1) / blockSize)
workBlock = int(gridSize / 2)
print("gridSize: ", gridSize)

def torch_go():
    output = torch.sum(input)
    return output
    
def cuda_go():
    cuda_module.reduce_forward(doutput, input, N)
    return torch.sum(doutput[:workBlock])

class Test_reduce:
    def test_torch(self):
        torch_res = show_time(torch_go, cuda=False, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, torch_res)
    def test_cuda(self):
        cuda_res = show_time(cuda_go, cuda=True, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, cuda_res)
    
if __name__ == '__main__':
    pytest.main(['-v', '-s'])
    print()