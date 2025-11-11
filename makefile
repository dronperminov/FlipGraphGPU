CXX = nvcc
FLAGS = -O3 -Xptxas -O3
OBJECTS = arg_parser.o addition.o flip_set.o scheme_integer.o scheme_z2.o flip_graph.o main.o
TARGET = flip_graph

all: $(OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) -o $(TARGET)

%.o: src/%.cu
	$(CXX) $(FLAGS) -I. -dc $< -o $@

clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f *.nsys-rep *.sqlite
