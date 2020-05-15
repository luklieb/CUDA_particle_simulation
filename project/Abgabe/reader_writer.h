#pragma once
#include "classes.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>



// Set according precision of "real" type
//#define convert std::stod
#define convert std::stof

class ParameterReader {
    private:
        std::map<std::string, std::string> data;
        // Check for key and return it
        std::string get_value(const std::string &key);
    public:
        // Read data
        void read(const std::string &filename);
        // Get parameters
        Params get();
};

class ParticleReader {
    private:
        std::vector<Particle> particles;
    public:
        // Read particles and return number of particles
        int read(const Params &parameters);
        // Get particles
        std::vector<Particle> get();
};

class OutputWriter {
    public:
        void write_vtk(const std::vector<Particle> &particles, const Params &params, size_t iter_v);
        void write_part(const std::vector<Particle> &particles, const Params &params, size_t iter_p);
};
