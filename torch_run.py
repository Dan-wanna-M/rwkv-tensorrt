import os,types
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import numpy as np
args = types.SimpleNamespace()
import torch
import tqdm
import timeit
from utils import SAMPLER
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

args.strategy = 'cuda fp16'

model = RWKV(model="/home/guest2/rwkv-onnx/RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth", strategy=args.strategy)
tokenizer = PIPELINE(model, f"20B_tokenizer.json")
sampler = SAMPLER("nucleus", 1.0, 0.7, 0.4, 0.4, 0.4, 256)
state = None


prompt = tokenizer.encode("User: Please describe an apple? Bot: Sure! an apple is")
logits, state = model.forward(prompt[:-1], state)
print("prompt loaded.")
temp = []
def generate(state):
  for i in range(256):
      temp.append(prompt[-1])
      logits, state = model.forward(temp, state)
      prompt.append(sampler.sample_nucleus(logits))
      temp.pop()

print(timeit.timeit(lambda: generate(state), number=5)/5)