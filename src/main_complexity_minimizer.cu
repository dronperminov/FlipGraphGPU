#include <iostream>
#include <ctime>

#include "entities/arg_parser.cuh"
#include "entities/complexity_minimizer.cuh"

int main(int argc, char* argv[]) {
    ArgParser parser("flip_graph", "Find best complexity of the fast matrix multiplication scheme in parallel using CUDA");

    parser.add("--schemes", ArgType::Natural, "INT", "number of schemes", "1024");
    parser.add("--max-iterations", ArgType::Natural, "INT", "number of flips per iterations", "10000");
    parser.add("--path", ArgType::String, "PATH", "path to save schemes", "schemes");
    parser.add("--input-path", ArgType::String, "PATH", "path to init scheme", "");
    parser.add("--block-size", ArgType::Natural, "INT", "number of cuda threads", "32");
    parser.add("--target-complexity", ArgType::Natural, "INT", "target complexity", "0");
    parser.add("--seed", ArgType::Natural, "INT", "random seed", "0");

    if (!parser.parse(argc, argv))
        return 0;

    int schemesCount = std::stoi(parser.get("--schemes"));
    int maxIterations = std::stoi(parser.get("--max-iterations"));
    std::string path = parser.get("--path");
    std::string inputPath = parser.get("--input-path");
    int blockSize = std::stoi(parser.get("--block-size"));
    int targetComplexity = std::stoi(parser.get("--target-complexity"));
    int seed = std::stoi(parser.get("--seed"));

    if (seed == 0)
        seed = time(0);

    Scheme scheme;

    std::ifstream f(inputPath);
    bool result = scheme.read(f);
    f.close();

    if (!result)
        return -1;

    std::cout << "Start complexity minimization algorithm" << std::endl;
    std::cout << "- scheme size: " << scheme.n[0] << scheme.n[1] << scheme.n[2] << std::endl;
    std::cout << "- scheme rank: " << scheme.m << std::endl;
    std::cout << "- scheme complexity: " << scheme.getComplexity() << std::endl;
    std::cout << "- schemes count: " << schemesCount << std::endl;
    std::cout << "- max iterations: " << maxIterations << std::endl;
    std::cout << "- path: " << path << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;
    std::cout << "- target complexity: " << targetComplexity << std::endl;
    std::cout << "- seed: " << seed << std::endl;

    ComplexityMinimizer minimizer(schemesCount, blockSize, maxIterations, path, seed);

    try {
        minimizer.minimize(scheme, targetComplexity);
        std::cout << "Success!" << std::endl;
    }
    catch (std::runtime_error e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
