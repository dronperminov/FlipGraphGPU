CXX = nvcc
FLAGS = -O3 -Xptxas -O3
OBJECTS = scheme.o flip_graph.o main.o
TARGET = flip_graph

all: $(OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) -o $(TARGET)

%.o: src/%.cu
	$(CXX) $(FLAGS) -I. -dc $< -o $@

clean:
	rm -f $(TARGET)
	rm -f $(OBJECTS)
	rm -f *.nsys-rep *.sqlite
