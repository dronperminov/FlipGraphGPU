#include "utils.cuh"

std::string getKey(int n1, int n2, int n3, bool sorted, bool withZeros) {
    std::vector<int> n = {n1, n2, n3};

    if (sorted)
        std::sort(n.begin(), n.end());

    std::stringstream ss;
    if (withZeros)
        ss << std::setfill('0') << std::setw(2);

    ss << n[0] << "x";

    if (withZeros)
        ss << std::setfill('0') << std::setw(2);

    ss << n[1] << "x";

    if (withZeros)
        ss << std::setfill('0') << std::setw(2);

    ss << n[2];
    return ss.str();
}

std::string getKey(const Scheme &scheme, bool sorted, bool withZeros) {
    return getKey(scheme.n[0], scheme.n[1], scheme.n[2], sorted, withZeros);
}

std::string prettyTime(double elapsed) {
    std::stringstream ss;

    if (elapsed < 60) {
        ss << std::setprecision(3) << std::fixed << elapsed;
    }
    else {
        int seconds = int(elapsed + 0.5);
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;

        ss << std::setw(2) << std::setfill('0') << hours << ":";
        ss << std::setw(2) << std::setfill('0') << minutes << ":";
        ss << std::setw(2) << std::setfill('0') << (seconds % 60);
    }

    return ss.str();
}

std::string prettyInt(int value) {
    std::stringstream ss;

    if (value < 1000)
        ss << value;
    else if (value < 1000000)
        ss << std::setprecision(2) << std::fixed << (value / 1000.0) << "K";
    else
        ss << std::setprecision(2) << std::fixed << (value / 1000000.0) << "M";

    return ss.str();
}
