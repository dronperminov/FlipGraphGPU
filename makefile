CXX = nvcc
FLAGS = -O3 -Xptxas -O3
OBJECTS = src/entities/arg_parser.o src/utils/utils.o src/entities/addition.o src/entities/flip_set.o src/schemes/scheme_integer.o src/schemes/scheme_z2.o
FLIP_GRAPH_OBJECTS = src/entities/flip_graph.o src/main_flip_graph.o
COMPLEXITY_MINIMIZER_OBJECTS = src/entities/complexity_minimizer.o src/main_complexity_minimizer.o

all: flip-graph complexity-minimizer

flip-graph: $(OBJECTS) $(FLIP_GRAPH_OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) $(FLIP_GRAPH_OBJECTS) -o flip_graph

complexity-minimizer: $(OBJECTS) $(COMPLEXITY_MINIMIZER_OBJECTS)
	$(CXX) $(FLAGS) $(OBJECTS) $(COMPLEXITY_MINIMIZER_OBJECTS) -o complexity_minimizer

%.o: %.cu
	$(CXX) $(FLAGS) -I. -dc $< -o $@

clean:
	rm -f flip_graph complexity_minimizer
	rm -f $(OBJECTS) $(FLIP_GRAPH_OBJECTS) $(COMPLEXITY_MINIMIZER_OBJECTS)
	rm -f *.nsys-rep *.sqlite
