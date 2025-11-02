#include <iostream>
#include <ctime>

#include "flip_graph.cuh"

int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 10) {
        std::cout << "Invalid number of arguments (" << (argc - 1) << ")" << std::endl;
        std::cout << "Usage: ./flip_graph [n1] [n2] [n3] [targetRank] [schemes count = 1024] [max iterations = 10000] [path = schemes] [block size = 16] [seed = time(0)]" << std::endl;
        return 0;
    }

    int n1 = atoi(argv[1]);
    int n2 = atoi(argv[2]);
    int n3 = atoi(argv[3]);
    int targetRank = atoi(argv[4]);
    int schemesCount = argc > 5 ? atoi(argv[5]) : 1024;
    int maxIterations = argc > 6 ? atoi(argv[6]) : 10000;
    std::string path = argc > 7 ? argv[7] : "schemes";
    int blockSize = argc > 8 ? atoi(argv[8]) : 32;
    int seed = argc > 9 ? atoi(argv[9]) : time(0);

    int reduceStart = maxIterations;
    int initialRank = n1 * n2 * n3;

    std::cout << "Start flip graph algorithm" << std::endl;
    std::cout << "- n: " << n1 << " " << n2 << " " << n3 << std::endl;
    std::cout << "- target rank: " << targetRank << std::endl;
    std::cout << "- schemes count: " << schemesCount << std::endl;
    std::cout << "- max iterations: " << maxIterations << std::endl;
    std::cout << "- path: " << path << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;
    std::cout << "- seed: " << seed << std::endl;

    FlipGraph flipGraph(n1, n2, n3, initialRank, targetRank, schemesCount, blockSize, maxIterations, path, reduceStart, seed);

    try {
        flipGraph.run();
        std::cout << "Success!" << std::endl;
    }
    catch (std::runtime_error e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
