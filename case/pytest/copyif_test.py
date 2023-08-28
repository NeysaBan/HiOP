import yaml
import torch
import pytest
from rectime import show_time
from torch.utils.cpp_extension import load


with open("case/pytest/pyconfig.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
device = 'cuda:' + str(config['device_id'])

cuda_module = load(name="copyif",
                        extra_include_paths=["include"],
                        sources=["reg_torch/copyif_op.cpp", "kernel/copyif_kernel.cu"],
                        build_directory="./build/",
                        verbose=True)

## test data
N = eval(config['N'])
smCnt = config['smCnt']
warpSize = config['warpSize']
blockSize = config['blockSize']
threadPerMultip = config['threadPerMultip']
dataPerBlock = blockSize ## 每个block可以处理的数据

src = torch.randint(low = -10, high = 10, size=(N, ), dtype=torch.float32).to(device)
newLen = int((N + dataPerBlock - 1) / dataPerBlock) * dataPerBlock
ddst = torch.zeros_like(src).to(device)
nRes = -1
ground_truth = torch.masked_select(src, src > 0)

gridSize = min(int((N + blockSize - 1) / blockSize), 
                int(smCnt * threadPerMultip / blockSize * warpSize))
# gridSize = int((N + blockSize - 1) / blockSize)
print("gridSize: ", gridSize)

def torch_go():
    output = torch.masked_select(src, src > 0)
    return output
    
def cuda_go():
    global nRes
    nRes = cuda_module.copyif_forward(ddst, src, N, gridSize)
    return ddst

class Test_copyif:
    def test_torch(self):
        torch_res = show_time(torch_go, cuda=False, device=device, ntest=config['ntest'])
        assert torch.allclose(ground_truth, torch_res)
    def test_cuda(self):
        cuda_res = show_time(cuda_go, cuda=True, device=device, ntest=config['ntest'])
        ground_truth = torch.masked_select(src, src > 0)
        ground_truth = torch.sort(ground_truth).values
        if nRes != -1:
            cuda_res = torch.sort(cuda_res[:nRes]).values ## 区间为左闭右闭
        else :
            cuda_res = torch.tensor([])
        assert torch.allclose(ground_truth, cuda_res)
    
if __name__ == '__main__':
    
    # pytest.main(['-v', '-s', "case/pytest/reduce_test.py"])
    pytest.main(['-v', '-s'])