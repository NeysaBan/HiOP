# HiOP

## 🥹 Environments

NVIDIA Driver: 535.86.10

CUDA (runtime): 12.2

CUDA (Pytorch Toolkit): 11.0

Python: 3.8.5

PyTorch: 1.7.1+cu110

## 🫵🏼 Structure

```
├── build
│   ├── *.so
├── include
│   ├── cuda_config.h
│   └── reduce_arr.h
├── kernel
│   └── reduce_kernel.cu
└── reg_torch
│   └── reduce_op.cpp
├── pytest.ini
├── case
│   ├── model_test
│   │   └── reduce_model.py
│   └── pytest
│       ├── pyconfig.yaml
│       ├── rectime.py
│       └── reduce_test.py
```

## 🤪 How To Run Test Case

```shell
python unit_test/reduce_test.py
```

## 🥹 Naive Res

测试平台：NVIDIA A100 80GB PCIe（无负载）

### Reduce Res

| 向量维度 | Pytorch耗时(us) | CUDA Kernel耗时(us) @warp |
| -------- | --------------- | ------------------------- |
| $2^{23}$ | 42              | 26                        |
| $2^{24}$ | 93              | 62                        |
| $2^{25}$ | 107             | 65                        |
| $2^{26}$ | 204             | 111                       |


### copy-if

| 向量维度 | Pytorch耗时(us) | CUDA Kernel耗时(us) |
| -------- | --------------- | ------------------- |
| $2^{23}$ | 42              | 26                  |
| $2^{24}$ | 435             | 235                 |
| $2^{25}$ | 799             | 427                 |
| $2^{26}$ | 1570            | 806                 | 

### fp16 gelu

| 向量维度 | Pytorch耗时(us) | CUDA Kernel耗时(us) |
| -------- | --------------- | ------------------- |
| $2^{23}$ | 65              | 125                 |
| $2^{24}$ | 109             | 190                 |
| $2^{25}$ | 197             | 315                 |
| $2^{26}$ | 377             | 572                 | 


### 🥥 blog
[Pytorch 自定义算子 JIT编译 踩坑记录](https://blog.neysaban.one/article/287547ee-8082-49cb-a182-8fa995d0043c)