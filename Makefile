NVCC = nvcc
CXXFLAGS = -std=c++17

TARGET = batch_image_processing
SOURCE = batch_image_processing.cu

all:
	$(NVCC) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

run:
	./$(TARGET) input_ppm output_ppm 500
