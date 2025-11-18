#include "utils.cuh"

std::string getKey(int n1, int n2, int n3) {
    std::vector<int> n = {n1, n2, n3};
    std::sort(n.begin(), n.end());

    std::stringstream ss;
    ss << n[0] << n[1] << n[2];
    return ss.str();
}

std::string getKey(const Scheme &scheme) {
    return getKey(scheme.n[0], scheme.n[1], scheme.n[2]);
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
