import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import torch
from transformers import AutoTokenizer
from utils import SAMPLER
import tqdm
import timeit

class InterOp():
    RnnOnly = True
    def __init__(self):
        self.logits = np.empty(io_tensors[logits_name]["shape"], dtype=io_tensors[logits_name]["dtype"])
        # self.output_state = np.empty(io_tensors["output_state"]["shape"], dtype=io_tensors["output_state"]["dtype"])
        self.input_token = np.empty((1,), dtype=io_tensors["input_token"]["dtype"])

    def forward(self, input_token, input_state):
        self.input_token[0] = input_token
        cuda.memcpy_htod(io_tensors["input_token"]['gpu_data'], self.input_token)
        cuda.memcpy_htod(io_tensors["input_state"]['gpu_data'], input_state)
        bindings = [None for i in range(len(io_tensors))]
        for i in io_tensors:
            bindings[io_tensors[i]["id"]] = io_tensors[i]["gpu_data"]
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(self.logits, io_tensors[logits_name]['gpu_data'])
        cuda.memcpy_dtoh(input_state, io_tensors["output_state"]['gpu_data'])
        return self.logits, input_state
        

# Load the serialized engine file
path = "/home/guest2/rwkv-onnx/RWKV_32_2560_16.trt"

with open(path, "rb") as f:
    engine_data = f.read()
# Create a runtime object
runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))

# Deserialize the engine
engine = runtime.deserialize_cuda_engine(engine_data)

# Create an execution context
with engine.create_execution_context() as context:
    # Get the binding names
    io_tensors = {}
    for binding_idx in range(engine.num_bindings):
        binding_name = engine.get_tensor_name(binding_idx)
        if not engine.binding_is_input(binding_idx):
            if binding_name != "output_state":
                logits_name = binding_name # logits name may vary
        io_tensors[binding_name] = {"id":binding_idx}
    
    for i in io_tensors:
        io_tensors[i]["shape"] = context.get_tensor_shape(i)
        io_tensors[i]["dtype"] = np.dtype(trt.nptype(engine.get_tensor_dtype(i)))
        io_tensors[i]["gpu_data"] = cuda.mem_alloc(int(np.prod(io_tensors[i]["shape"]) * io_tensors[i]["dtype"].itemsize))
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    embed = int(path.split("_")[2].split(".")[0])
    layers = int(path.split("_")[1])
    isUnsafeWKV = io_tensors["input_state"]["shape"][0] < 5 * layers
    model = InterOp()
    print(f"isUnsafeWKV: {isUnsafeWKV}")
    sampler = SAMPLER("nucleus", 1.0, 0.7, 0.4, 0.4, 0.4, 256)
    state = np.array((([[0.01] * embed, [0.01] * embed, [0.01] * embed, [
        0.01] * embed] + ([[-1e30] * embed] if not isUnsafeWKV else []))) * layers, dtype=io_tensors["input_state"]["dtype"])
    prompt = tokenizer.encode("User: Please describe an apple? Bot: Sure! an apple is")
    for token in tqdm.tqdm(prompt[:-1]):
        logits, state = model.forward(token, state)
        assert not np.isnan(state).any(), str(state)
    print("prompt loaded.")
    def generate(state):
      for i in range(256):
          logits, state = model.forward(prompt[-1], state)
          prompt.append(sampler.sample_nucleus(torch.from_numpy(logits)))
          # print(tokenizer.decode(prompt[-1]),end="", flush=True)
      # print(tokenizer.decode(prompt))
    
    print(timeit.timeit(lambda: generate(state), number=5)/5)