#include <iostream>
#include <ctime>

#include "flip_graph.cuh"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 7) {
        std::cout << "Invalid number of arguments (" << (argc - 1) << ")" << std::endl;
        std::cout << "Usage: ./flip_graph [n] [targetRank] [schemes count = 1024] [max iterations = 10000] [path = schemes] [block size = 16]" << std::endl;
        return 0;
    }

    int n = atoi(argv[1]);
    int targetRank = atoi(argv[2]);
    int schemesCount = argc > 3 ? atoi(argv[3]) : 1024;
    int maxIterations = argc > 4 ? atoi(argv[4]) : 10000;
    std::string path = argc > 5 ? argv[5] : "schemes";
    int blockSize = argc > 6 ? atoi(argv[6]) : 32;
    int reduceStart = maxIterations;
    int seed = 42;//time(0);

    int initialRank = n*n*n;

    std::cout << "Start flip graph algorithm" << std::endl;
    std::cout << "- n: " << n << std::endl;
    std::cout << "- target rank: " << targetRank << std::endl;
    std::cout << "- schemes count: " << schemesCount << std::endl;
    std::cout << "- max iterations: " << maxIterations << std::endl;
    std::cout << "- path: " << path << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;
    std::cout << "- seed: " << seed << std::endl;
    // std::cout << "- reduce probability: " << (reduceProbability * 100) << "%" << std::endl << std::endl;

    FlipGraph flipGraph(n, initialRank, targetRank, schemesCount, blockSize, maxIterations, path, reduceStart, seed);
    flipGraph.run();

    std::cout << "Success!" << std::endl;
    return 0;
}
