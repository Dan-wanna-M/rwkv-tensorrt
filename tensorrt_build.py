import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
input_path = "/home/guest2/rwkv-onnx/RWKV_32_2560_32.onnx"

with trt.OnnxParser(network, TRT_LOGGER) as parser:
    # Load the ONNX model file
    with open(input_path, 'rb') as model_file:
        parser.parse(model_file.read())

# Configure builder settings, optimizations, and desired precision (FP16, INT8, etc.)

# Build the TensorRT engine
config = builder.create_builder_config()
config.flags = 1<<int(trt.BuilderFlag.FP16)
config.builder_optimization_level = 5
engine_data = builder.build_serialized_network(network, config)

# Serialize the engine to a file
with open(f"/home/guest2/rwkv-onnx/RWKV_32_2560_16.trt", 'wb') as engine_file:
    engine_file.write(engine_data)
