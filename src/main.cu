#include <iostream>
#include <ctime>

#include "flip_graph.cuh"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 8) {
        std::cout << "Invalid number of arguments (" << (argc - 1) << ")" << std::endl;
        std::cout << "Usage: ./flip_graph [n] [targetRank] [schemes count = 1024] [max iterations = 10000] [path = schemes] [block size = 16] [seed = time(0)]" << std::endl;
        return 0;
    }

    int n = atoi(argv[1]);
    int targetRank = atoi(argv[2]);
    int schemesCount = argc > 3 ? atoi(argv[3]) : 1024;
    int maxIterations = argc > 4 ? atoi(argv[4]) : 10000;
    std::string path = argc > 5 ? argv[5] : "schemes";
    int blockSize = argc > 6 ? atoi(argv[6]) : 32;
    int reduceStart = maxIterations;
    int seed = argc > 7 ? atoi(argv[7]) : time(0);

    int initialRank = n*n*n;

    std::cout << "Start flip graph algorithm" << std::endl;
    std::cout << "- n: " << n << std::endl;
    std::cout << "- target rank: " << targetRank << std::endl;
    std::cout << "- schemes count: " << schemesCount << std::endl;
    std::cout << "- max iterations: " << maxIterations << std::endl;
    std::cout << "- path: " << path << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;
    std::cout << "- seed: " << seed << std::endl;

    FlipGraph flipGraph(n, initialRank, targetRank, schemesCount, blockSize, maxIterations, path, reduceStart, seed);

    try {
        flipGraph.run();
        std::cout << "Success!" << std::endl;
    }
    catch (std::runtime_error e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
