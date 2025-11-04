#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

enum class ArgType {
    String, Natural
};

struct Arg {
    std::string name;
    ArgType type;
    std::string meta;
    std::string description;
    std::string value;
};

class ArgParser {
    std::string name;
    std::string description;
    std::vector<Arg> args;
    std::unordered_map<std::string, std::string> parsed;
    int required;

    bool isNatural(const std::string &value) const;
    bool validate(const Arg &arg, const std::string &value) const;
    bool checkRequired() const;
public:
    ArgParser(const std::string &name, const std::string &description = "");

    void add(const std::string &name, ArgType type, const std::string &meta, const std::string &description = "", const std::string &value = "");
    bool parse(int argc, char *argv[]);
    void help();

    std::string get(const std::string &name) const;
};
