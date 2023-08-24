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

‘’‘
python unit_test/reduce_test.py
’‘’
