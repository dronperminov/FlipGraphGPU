CXX = nvcc
FLAGS = -O3 -Xptxas -O3
OBJECTS = arg_parser.o utils.o addition.o flip_set.o scheme_integer.o scheme_z2.o
TARGET = flip_graph

all: flip-graph complexity-minimizer

flip-graph: $(OBJECTS) flip_graph.o main_flip_graph.o
	$(CXX) $(FLAGS) $(OBJECTS) flip_graph.o main_flip_graph.o -o flip_graph

complexity-minimizer: $(OBJECTS) complexity_minimizer.o main_complexity_minimizer.o
	$(CXX) $(FLAGS) $(OBJECTS) complexity_minimizer.o main_complexity_minimizer.o -o complexity_minimizer

%.o: src/%.cu
	$(CXX) $(FLAGS) -I. -dc $< -o $@

clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f *.nsys-rep *.sqlite
