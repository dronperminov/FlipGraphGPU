#include "arg_parser.cuh"

ArgParser::ArgParser(const std::string &name, const std::string &description) {
    this->name = name;
    this->description = description;
    this->required = 0;
}

void ArgParser::add(const std::string &name, ArgType type, const std::string &meta, const std::string &description, const std::string &value) {
    args.push_back({name, type, meta, description, value});

    if (value == "")
        required++;
}

bool ArgParser::parse(int argc, char *argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        help();
        return false;
    }

    if (argc % 2 != 1) {
        std::cerr << "Invalid number of arguments" << std::endl;
        help();
        return false;
    }

    std::unordered_map<std::string, Arg> name2arg;

    for (auto arg : args) {
        name2arg[arg.name] = arg;

        if (arg.value != "")
            parsed[arg.name] = arg.value;
    }

    for (int i = 1; i < argc; i += 2) {
        std::string name = argv[i];

        if (name2arg.find(name) == name2arg.end()) {
            std::cerr << "unknown argument \"" << name << "\"" << std::endl;
            return false;
        }

        if (i == argc - 1) {
            std::cerr << "no value for arg \"" << name << "\"" << std::endl;
            return false;
        }

        parsed[name] = argv[i + 1];

        if (!validate(name2arg[name], parsed[name]))
            return false;
    }

    return checkRequired();
}

void ArgParser::help() {
    std::cout << description << std::endl;
    std::cout << "Usage: ./" << name;

    for (auto arg : args) {
        if (arg.value != "")
            std::cout << " [" << arg.name << " " << arg.value << "]";
        else
            std::cout << " " << arg.name << " " << arg.meta;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Arguments description:" << std::endl;

    for (auto arg : args) {
        std::cout << arg.name << ": " << arg.description;

        if (arg.value != "")
            std::cout << " (default: " << arg.value << ")";

        std::cout << std::endl;
    }
}

std::string ArgParser::get(const std::string &name) const {
    auto it = parsed.find(name);
    if (it == parsed.end())
        throw std::runtime_error("no parsed value for argument \"" + name + "\"");

    return it->second;
}

bool ArgParser::validate(const Arg &arg, const std::string &value) const {
    if (arg.type == ArgType::String)
        return true;

    if (arg.type == ArgType::Natural && !isNatural(value)) {
        std::cerr << "value for arg \"" << arg.name << "\" is not natural (" << value << ")" << std::endl;
        return false;
    }

    if (arg.type == ArgType::Real && !isReal(value)) {
        std::cerr << "value for arg \"" << arg.name << "\" is not real (" << value << ")" << std::endl;
        return false;
    }

    return true;
}

bool ArgParser::isNatural(const std::string &value) const {
    for (size_t i = 0; i < value.size(); i++)
        if (value[i] < '0' || value[i] > '9')
            return false;

    return true;
}

bool ArgParser::isReal(const std::string &value) const {
    if (value.size() == 0)
        return false;

    size_t start = value[0] == '-' ? 1 : 0;
    bool point = false;

    for (size_t i = start; i < value.size(); i++) {
        if (value[i] == '.') {
            if (point)
                return false;

            point = true;
        }
        else if (value[i] < '0' || value[i] > '9')
            return false;
    }

    return true;
}

bool ArgParser::checkRequired() const {
    for (auto arg : args) {
        if (arg.value == "" && parsed.find(arg.name) == parsed.end()) {
            std::cerr << "no value for argument \"" << arg.name << "\"" << std::endl;
            return false;
        }
    }

    return true;
}
