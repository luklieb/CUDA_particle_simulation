#pragma once
#include <map>
#include <iostream>
#include <fstream>
#include <string>

//define FLOAT or DOUBLE
#define DOUBLE

#ifdef FLOAT
typedef float real;
#define convert std::stof
#else
typedef double re/Users/lukas/Documents/Universitaet/Studium/SS17/HESPA/2_Exercise/Abgabe/MDClasses.hal;
#define convert std::stod
#endif

struct Params {
    std::string input;
    int block_size;
    real timestep_length;
    real time_end;
    real epsilon;
    real sigma;
    int part_out_freq;
    std::string part_output;
    int vtk_out_freq;
    std::string vtk_output;
	int num_part;
};

class ParameterReader {
    private:
        std::map<std::string,std::string> data;

    public:
        // Read data
        bool read(const std::string& filename) {
            std::ifstream parameters (filename);
            std::string line;
            if (parameters.is_open()) {
                while(!parameters.eof()){
                    std::getline(parameters, line);
                    if((line.length() <= 1)) continue;
                    size_t pos = line.find_first_of(" ");
                    size_t last = line.find_last_of(" ");
                    data.insert(std::pair<std::string,std::string>(line.substr(0, pos), line.substr(last+1, line.length())));
                }
            }
            parameters.close();
            return true;
        }
        // Check for key
        inline bool is_defined(const std::string& key) const{
            if(data.count(key) == 0) return false;
            return true;
        }

        // Get Parameter
        Params get() {
            Params params;
            params.input = (data.find("part_input_file")->second);
            params.block_size = std::stoi(data.find("cl_workgroup_1dsize")->second);
            params.timestep_length = convert(data.find("timestep_length")->second);
            params.time_end = convert(data.find("time_end")->second);
            params.epsilon = convert(data.find("epsilon")->second);
            params.sigma = convert(data.find("sigma")->second);
            params.part_out_freq = std::stoi(data.find("part_out_freq")->second);
            params.part_output = (data.find("part_out_name_base")->second);
            params.vtk_out_freq = std::stoi(data.find("vtk_out_freq")->second);
            params.vtk_output = (data.find("vtk_out_name_base")->second);
            return params;
        }
};

struct Particle {
    real m;
    real x0;
    real x1;
    real x2;
    real v0;
    real v1;
    real v2;
    real force0;
    real force1;
    real force2;
    real force0_old;
    real force1_old;
    real force2_old;
};
