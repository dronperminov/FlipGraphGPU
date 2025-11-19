#include <iostream>
#include <ctime>

#include "entities/arg_parser.cuh"
#include "entities/flip_graph.cuh"

int main(int argc, char* argv[]) {
    ArgParser parser("flip_graph", "Find fast matrix multiplication in parallel using CUDA");

    parser.add("-n1", ArgType::Natural, "INT", "number of first matrix rows");
    parser.add("-n2", ArgType::Natural, "INT", "number of first matrix columns and second matrix rows");
    parser.add("-n3", ArgType::Natural, "INT", "number of second matrix columns");
    parser.add("--schemes", ArgType::Natural, "INT", "number of schemes", "1024");
    parser.add("--max-iterations", ArgType::Natural, "INT", "number of flips per iterations", "10000");
    parser.add("--path", ArgType::String, "PATH", "path to save schemes", "schemes");
    parser.add("--input-path", ArgType::String, "PATH", "path to init schemes", "null");
    parser.add("--block-size", ArgType::Natural, "INT", "number of cuda threads", "32");
    parser.add("--expand-probability", ArgType::Real, "REAL", "expand edge probability (divided by max iterations)", "0.01");
    parser.add("--reduce-probability", ArgType::Real, "REAL", "reduce edge probability (divided by max iterations)", "0.0");
    parser.add("--sandwiching-probability", ArgType::Real, "REAL", "sandwiching edge probability (divided by max iterations)", "0.0");
    parser.add("--basis-probability", ArgType::Real, "REAL", "basis edge probability (divided by max iterations)", "0.0");
    parser.add("--resize-probability", ArgType::Real, "REAL", "project/extend edge probability", "0.2");
    parser.add("--seed", ArgType::Natural, "INT", "random seed", "0");

    if (!parser.parse(argc, argv))
        return 0;

    int n1 = std::stoi(parser.get("-n1"));
    int n2 = std::stoi(parser.get("-n2"));
    int n3 = std::stoi(parser.get("-n3"));
    int schemesCount = std::stoi(parser.get("--schemes"));
    int maxIterations = std::stoi(parser.get("--max-iterations"));
    std::string path = parser.get("--path");
    std::string inputPath = parser.get("--input-path");
    int blockSize = std::stoi(parser.get("--block-size"));
    int seed = std::stoi(parser.get("--seed"));

    FlipGraphProbabilities probabilities;
    probabilities.expand = std::stod(parser.get("--expand-probability"));
    probabilities.reduce = std::stod(parser.get("--reduce-probability"));
    probabilities.sandwiching = std::stod(parser.get("--sandwiching-probability"));
    probabilities.basis = std::stod(parser.get("--basis-probability"));
    probabilities.resize = std::stod(parser.get("--resize-probability"));

    if (seed == 0)
        seed = time(0);

    if (n1 * n2 > MAX_MATRIX_ELEMENTS || n2 * n3 > MAX_MATRIX_ELEMENTS || n1 * n3 > MAX_MATRIX_ELEMENTS) {
        std::cout << "Error sizes, please increase MAX_MATRIX_ELEMENTS (now: " << MAX_MATRIX_ELEMENTS << ")" << std::endl;
        return 0;
    }

    if (n1 * n2 * n3 > MAX_RANK) {
        std::cout << "Error sizes, please increase MAX_RANK (now: " << MAX_RANK << ")" << std::endl;
        return 0;
    }

    std::cout << "Start flip graph algorithm" << std::endl;
    std::cout << "- n: " << n1 << " " << n2 << " " << n3 << std::endl;
    std::cout << "- schemes count: " << schemesCount << std::endl;
    std::cout << "- max iterations: " << maxIterations << std::endl;
    std::cout << "- path: " << path << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;
    std::cout << "- probabilities:" << std::endl;
    std::cout << "  - expand: " << probabilities.expand << std::endl;
    std::cout << "  - reduce: " << probabilities.reduce << std::endl;
    std::cout << "  - sandwiching: " << probabilities.sandwiching << std::endl;
    std::cout << "  - basis: " << probabilities.basis << std::endl;
    std::cout << "  - resize: " << probabilities.resize << std::endl;
    std::cout << "- seed: " << seed << std::endl;

    FlipGraph flipGraph(n1, n2, n3, schemesCount, blockSize, maxIterations, path, probabilities, seed);

    if (inputPath != "null") {
        std::ifstream f(inputPath);
        bool result = flipGraph.initializeFromFile(f);
        f.close();

        if (!result)
            return -1;
    }
    else {
        flipGraph.initializeNaive();
    }

    try {
        flipGraph.run();
        std::cout << "Success!" << std::endl;
    }
    catch (std::runtime_error e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
