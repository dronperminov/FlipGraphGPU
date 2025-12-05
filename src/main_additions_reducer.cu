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
    parser.add("--schemes-count", ArgType::Natural, "INT", "number of parallel schemes", "1");
    parser.add("--block-size", ArgType::Natural, "INT", "number of cuda threads", "32");
    parser.add("--start-additions", ArgType::Natural, "INT", "upper bound of additions for check optimality", "0");
    parser.add("--max-flips", ArgType::Natural, "INT", "number of scheme flips", "0");
    parser.add("--max-no-improvements", ArgType::Natural, "INT", "max iterations without improvements", "3");
    parser.add("--seed", ArgType::Natural, "INT", "random seed", "0");

    if (!parser.parse(argc, argv))
        return 0;

    std::string inputPath = parser.get("-i");
    std::string outputPath = parser.get("-o");

    int count = std::stoi(parser.get("--count"));
    int schemesCount = std::stoi(parser.get("--schemes-count"));
    int blockSize = std::stoi(parser.get("--block-size"));
    int maxNoImprovements = std::stoi(parser.get("--max-no-improvements"));
    int startAdditions = std::stoi(parser.get("--start-additions"));
    int maxFlips = std::stoi(parser.get("--max-flips"));
    int seed = std::stoi(parser.get("--seed"));

    if (seed == 0)
        seed = time(0);

    std::ifstream f(inputPath);
    if (!f) {
        std::cout << "Unable to open file \"" << inputPath << "\"" << std::endl;
        return -1;
    }

    std::cout << "Compiled configuration:" << std::endl;
    std::cout << "- max expressions count (U / V / W): " << MAX_U_EXPRESSIONS << " / " << MAX_V_EXPRESSIONS << " / " << MAX_W_EXPRESSIONS << std::endl;
    std::cout << "- max real variables (U / V / W): " << MAX_U_REAL_VARIABLES << " / " << MAX_V_REAL_VARIABLES << " / " << MAX_W_REAL_VARIABLES << std::endl;
    std::cout << "- max subexpressions (U / V / W): " << MAX_U_SUBEXPRESSIONS << " / " << MAX_V_SUBEXPRESSIONS << " / " << MAX_W_SUBEXPRESSIONS << std::endl;
    std::cout << "- max fresh variables (U / V / W): " << MAX_U_FRESH_VARIABLES << " / " << MAX_V_FRESH_VARIABLES << " / " << MAX_W_FRESH_VARIABLES << std::endl;
    std::cout << std::endl;

    std::cout << "Start additions reduction algorithm" << std::endl;
    std::cout << "- count: " << count << std::endl;
    std::cout << "- schemes count: " << schemesCount << std::endl;
    std::cout << "- output path: " << outputPath << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;

    if (startAdditions > 0)
        std::cout << "- start additions: " << startAdditions << std::endl;

    if (maxFlips > 0)
        std::cout << "- max flips: " << maxFlips << std::endl;

    std::cout << "- max no improvements: " << maxNoImprovements << std::endl;
    std::cout << "- seed: " << seed << std::endl;
    std::cout << std::endl;

    SchemeAdditionsReducer reducer(count, schemesCount, maxFlips, seed, blockSize, outputPath);
    bool correct = reducer.read(f);
    f.close();

    if (!correct)
        return -1;

    reducer.reduce(maxNoImprovements, startAdditions);
    return 0;
}
