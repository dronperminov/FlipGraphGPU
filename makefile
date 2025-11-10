CXX = nvcc
FLAGS = -O3 -Xptxas -O3
OBJECTS_Z2 = arg_parser.o flip_set.o scheme_z2.o flip_graph.o main.o
OBJECTS_INTEGER = arg_parser.o addition.o flip_set.o scheme_integer.o flip_graph.o main.o
TARGET = flip_graph

z2: $(OBJECTS_Z2)
	$(CXX) $(FLAGS) $(OBJECTS_Z2) -o $(TARGET)

integer: $(OBJECTS_INTEGER)
	$(CXX) $(FLAGS) $(OBJECTS_INTEGER) -o $(TARGET)

%.o: src/%.cu
	$(CXX) $(FLAGS) -I. -dc $< -o $@

clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f *.nsys-rep *.sqlite
