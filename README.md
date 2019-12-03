# Benchmark of TVM quantized model on CUDA
This repository contains benchmark code of int8 inference speed of TVM for the blog post [Automating Optimization of Quantized Deep Learning Models on CUDA](https://tvm.ai/2019/04/29/opt-cuda-quantized.html). The benchmark of MXNet and TensorRT is provided as baseline.

## How to Run
#### TVM
The benchmark is conducted using [tvm@e22b58](https://github.com/dmlc/tvm/tree/e22b5802a3e6c269d76e52428ca81cbd4b7d8304). (**This is an outdated version. Please checkout [this branch](https://github.com/vinx13/tvm-cuda-int8-benchmark/tree/latest) to run with recent tvm version.**)
LLVM and CUDA need to be enabled.
Compute Capability 6.1 CUDA device is required to support the `dp4a` instruction.

We only provide auto-tuning logs on NVIDIA GTX 1080. To run on other devices, you can follow the [AutoTVM tutorial](https://docs.tvm.ai/tutorials/autotvm/tune_relay_cuda.html) to run auto-tuning.

```
python3 run_tvm.py --log_file logs/history_best_1080.log
```

#### MXNet
MXNet 1.4, cuDNN 7.3+ are required.
```
python3 run_mxnet.py
```

#### TensorRT
TensorRT 5 is required. We use onnx models as input. The onnx models will be generated from MXNet when running the benchmark script.
```
cd TensorRT; make; cd -;
python3 run_tensorrt.py
```

## Result
![benchmark results](/image/benchmark.svg)

