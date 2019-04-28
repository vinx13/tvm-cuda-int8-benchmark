#include <assert.h>
#include <sys/stat.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>

#define CHECK(status)                                                 \
  do {                                                                \
    auto ret = (status);                                              \
    if (ret != 0) {                                                   \
      std::cout << __LINE__ << " Cuda failure: " << ret << std::endl; \
      abort();                                                        \
    }                                                                 \
  } while (0)

using namespace nvinfer1;

static const bool TEST_INT8 = true;

float inline randF() { return rand() / 65536.0f; }

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  Logger() : Logger(Severity::kWARNING) {}

  Logger(Severity severity) : reportableSeverity(severity) {}

  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportableSeverity{Severity::kWARNING};
};

static Logger gLogger;

class Calibrator : public nvinfer1::IInt8LegacyCalibrator {
 public:
  Calibrator(int batch_size, int inputSize, int firstBatch = 0,
             double cutoff = 0.5, double quantile = 0.5, bool readCache = true)
      : mFirstBatch(firstBatch), mReadCache(readCache), index(0) {
    using namespace nvinfer1;
    mInputCount = batch_size * inputSize;
    CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    reset(cutoff, quantile);
    mData.resize(mInputCount);
  }

  virtual ~Calibrator() { CHECK(cudaFree(mDeviceInput)); }

  int getBatchSize() const override { return 1; }
  double getQuantile() const override { return mQuantile; }
  double getRegressionCutoff() const override { return mCutoff; }

  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) override {
    for (auto& elem : mData) elem = randF();
    CHECK(cudaMemcpy(mDeviceInput, mData.data(), mInputCount,
                     cudaMemcpyHostToDevice));

    if (index++ > 0) {
      return false;
    }

    bindings[0] = mDeviceInput;
    return true;
  }

  const void* readCalibrationCache(size_t& length) override { return nullptr; }

  void writeCalibrationCache(const void* cache, size_t length) override {}

  const void* readHistogramCache(size_t& length) override {
    length = mHistogramCache.size();
    return length ? &mHistogramCache[0] : nullptr;
  }

  void writeHistogramCache(const void* cache, size_t length) override {
    mHistogramCache.clear();
    std::copy_n(reinterpret_cast<const char*>(cache), length,
                std::back_inserter(mHistogramCache));
  }

  void reset(double cutoff, double quantile) {
    mCutoff = cutoff;
    mQuantile = quantile;
  }

 private:
  int index;
  int mFirstBatch;
  std::vector<float> mData;
  double mCutoff, mQuantile;
  bool mReadCache{true};

  size_t mInputCount;
  void* mDeviceInput{nullptr};
  std::vector<char> mCalibrationCache, mHistogramCache;
};

ICudaEngine* onnxToTRTModel(
    const std::string& modelFile,  // name of the onnx model
    unsigned int maxBatchSize,
    DataType dataType,
    int* inputSize, int* outputSize) {
  // create the builder
  IBuilder* builder = createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

  if (!parser->parseFromFile(modelFile.c_str(),
                             static_cast<int>(ILogger::Severity::kWARNING))) {
    exit(EXIT_FAILURE);
  }

  auto inputDims = network->getInput(0)->getDimensions();
  *inputSize = *outputSize = 1;
  for (int i = 0; i < inputDims.nbDims; i++) {
    *inputSize *= inputDims.d[i];
  }
  auto outputDims = network->getOutput(0)->getDimensions();
  for (int i = 0; i < outputDims.nbDims; i++) {
    *outputSize *= outputDims.d[i];
  }

  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 30);
  builder->setAverageFindIterations(1);
  builder->setMinFindIterations(1);
  builder->setInt8Mode(true);
  Calibrator calibrator(1, *inputSize);
  builder->setInt8Calibrator(&calibrator);

  ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);

  // we don't need the network any more, and we can destroy the parser
  network->destroy();
  parser->destroy();
  builder->destroy();

  return engine;
}

float doInference(IExecutionContext& context, float* input, int batchSize,
                  int repeat, int inputSize, int outputSize) {
  const ICudaEngine& engine = context.getEngine();
  // input and output buffer pointers that we pass to the engine - the engine
  // requires exactly IEngine::getNbBindings(), of these, but in this case we
  // know that there is exactly one input and one output.
  assert(engine.getNbBindings() == 2);
  void* buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  int inputIndex, outputIndex;
  for (int b = 0; b < engine.getNbBindings(); ++b) {
    if (engine.bindingIsInput(b))
      inputIndex = b;
    else
      outputIndex = b;
  }

  // create GPU buffers and a stream
  CHECK(
      cudaMalloc(&buffers[inputIndex], batchSize * inputSize * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex],
                   batchSize * outputSize * sizeof(float)));

  CHECK(cudaMemcpy(buffers[inputIndex], input,
                   batchSize * inputSize * sizeof(float),
                   cudaMemcpyHostToDevice));
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
  CHECK(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));

  nvtxRangeId_t id1 = nvtxRangeStartA("Running");
  cudaEventRecord(start, stream);

  for (int i = 0; i < repeat; i++) {
    context.enqueue(batchSize, buffers, stream, nullptr);
  }
  cudaEventRecord(stop, stream);
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaStreamSynchronize(stream));
  nvtxRangeEnd(id1);

  float elapsedTime;
  CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  // release the stream and the buffers
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
  return elapsedTime;
}

int main(int argc, char** argv) {
  // create a TensorRT model from the onnx model and serialize it to a stream
  if (argc < 3) {
    std::cout << argv[0] << " model_name batch_size" << std::endl;
    exit(EXIT_FAILURE);
  }
  int repeat = 1000;
  if (argc == 4) {
    repeat = std::atoi(argv[3]);
  }

  std::string modelName = argv[1];
  int batchSize = std::atoi(argv[2]);

  auto dtype = TEST_INT8 ? DataType::kINT8 : DataType::kFLOAT;

  int inputSize, outputSize;
  auto engine =
      onnxToTRTModel(modelName, batchSize, dtype, &inputSize, &outputSize);

  std::vector<float> data(batchSize * inputSize);
  for (auto& elem : data) {
    elem = randF();
  }

  // deserialize the engine
  IRuntime* runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);

  // run inference
  auto time = doInference(*context, data.data(), batchSize, repeat, inputSize,
                          outputSize);

  std::cout << "TensorRT " << modelName << ' ' << "(batch=" << batchSize << ") "
            << (TEST_INT8 ? "int8" : "fp32");
  std::cout << " Avg. Time: " << time / repeat << "ms" << std::endl;

  // destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
