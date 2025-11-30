#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

#include "entities/arg_parser.cuh"
#include "entities/scheme_additions_reducer.cuh"

int main(int argc, char* argv[]) {
    ArgParser parser("addition_reducer", "Find best additions number of the fast matrix multiplication scheme in parallel using CUDA");

    parser.add("-i", ArgType::String, "PATH", "path to init scheme(s)", "");
    parser.add("-o", ArgType::String, "PATH", "path to save schemes", "schemes/additions_reduced");
    parser.add("--count", ArgType::Natural, "INT", "number of parallel reducers", "32");
    parser.add("--block-size", ArgType::Natural, "INT", "number of cuda threads", "32");
    parser.add("--max-no-improvements", ArgType::Natural, "INT", "max iterations without improvements", "3");
    parser.add("--seed", ArgType::Natural, "INT", "random seed", "0");

    if (!parser.parse(argc, argv))
        return 0;

    std::string inputPath = parser.get("-i");
    std::string outputPath = parser.get("-o");

    int count = std::stoi(parser.get("--count"));
    int blockSize = std::stoi(parser.get("--block-size"));
    int maxNoImprovements = std::stoi(parser.get("--max-no-improvements"));
    int seed = std::stoi(parser.get("--seed"));

    if (seed == 0)
        seed = time(0);

    std::ifstream f(inputPath);
    if (!f) {
        std::cout << "Unable to open file \"" << inputPath << "\"" << std::endl;
        return -1;
    }

    std::cout << "Start additions reduction algorithm" << std::endl;
    std::cout << "- count: " << count << std::endl;
    std::cout << "- output path: " << outputPath << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;
    std::cout << "- max no improvements: " << maxNoImprovements << std::endl;
    std::cout << "- seed: " << seed << std::endl;
    std::cout << std::endl;

    SchemeAdditionsReducer reducer(count, seed, blockSize, outputPath);
    bool correct = reducer.read(f);
    f.close();

    if (!correct) {
        std::cout << "Error during scheme reading" << std::endl;
        return -1;
    }

    reducer.reduce(maxNoImprovements);
    return 0;
}
