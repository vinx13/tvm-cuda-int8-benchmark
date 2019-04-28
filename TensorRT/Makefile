CXXFLAGS=-std=c++11
LIBS=-lcudart -lnvinfer -lnvparsers -lnvToolsExt -lnvonnxparser
bench: bench.cc
	g++ $^ $(CXXFLAGS) -o $@ $(LIBS)

