from typing import Any
import torch
import argparse
import numpy as np
from rectime import show_time
from torch.autograd import Function
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description='special config')
parser.add_argument('--device', dest='device', type=str, default="1", required = False, help='idx of cuda')
parser.add_argument('--ntest', dest='ntest', type=int, default=10, required = False, help='iter of test')
parser.add_argument('--blockSize', dest='blockSize', type=int, default=256, required = False, help='thread num in a block')
args = parser.parse_args()

args.device = 'cuda:' + args.device

cuda_module = load(name="reduce_arr",
                        extra_include_paths=["include"],
                        sources=["reg_torch/reduce_op.cpp", "kernel/reduce_kernel.cu"],
                        verbose=True)

N = 100000
# input = torch.randint(low=0, high=10, size=(1, N), dytype=torch.int32, device=args.device)
input = torch.rand(size=(1, N), dtype=torch.float32).to(args.device)
output = torch.empty(1).to(args.device)
doutput = torch.empty(1).to(args.device)

# class cudaReduce(Function):
#     @staticmethod
#     def forward(ctx, output, input, n, device):

#         c = torch.empty(n).to(device)
#         cuda_module.torch_launch_reduce(output, input, n)

#         return c 

#     @staticmethod
#     def backward(ctx: Any, *grad_outputs: Any) -> Any:

#         return super().backward(ctx, *grad_outputs)

def cuda_go():
    cuda_module.torch_launch_reduce(doutput, input, N)

def torch_go():
    output = torch.sum(input)

show_time(cuda_go, cuda=True, device=args.device, ntest=args.ntest)

show_time(torch_go, cuda=False, device=args.device, ntest=args.ntest)

torch.allclose(output, doutput)
print("Kernel test passed.")