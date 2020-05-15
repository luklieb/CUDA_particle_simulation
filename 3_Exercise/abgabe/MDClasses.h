#pragma once
#include <map>
#include <iostream>
#include <fstream>
#include <string>

typedef double real;
#define convert std::stod

struct Params {
    std::string input;
    int block_size;
    int block_size_x;
    int block_size_y;
    int block_size_z;
    real timestep_length;
    real time_end;
    real epsilon;
    real sigma;
    int part_out_freq;
    std::string part_output;
    int vtk_out_freq;
    std::string vtk_output;
    int num_part;
    real x_min;
    real x_max;
    real y_min;
    real y_max;
    real z_max;
    real z_min;
    int x_n; //anzahl zellen in x Richtung
    int y_n;
    int z_n;
    real r_cut;
    real x_len; //zell laenge in x Richtung
    real y_len;
    real z_len;
    real nl_vel; //max vel for neighbour approach
    int nl_size;
    int nl_freq;
};

std::string get_value(const std::map<std::string, std::string> & d, const std::string & s){
    if(d.find(s) != d.end()){
        return d.find(s)->second;
    }
    std::cerr << "falscher key in get_value" << std::endl;
    return "-1";
}

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
                    size_t ret = line.find_first_of("\n\r");
                    data.insert(std::pair<std::string,std::string>(line.substr(0, pos), line.substr(last+1, ret-last-1)));
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
            params.block_size_x = std::stoi(get_value(data, "cl_workgroup_3dsize_x"));
            params.block_size_y = std::stoi(get_value(data, "cl_workgroup_3dsize_y"));
            params.block_size_z = std::stoi(get_value(data, "cl_workgroup_3dsize_z"));
            params.timestep_length = convert(data.find("timestep_length")->second);
            params.time_end = convert(data.find("time_end")->second);
            params.epsilon = convert(data.find("epsilon")->second);
            params.sigma = convert(data.find("sigma")->second);
            params.part_out_freq = std::stoi(data.find("part_out_freq")->second);
            params.part_output = (data.find("part_out_name_base")->second);
            params.vtk_out_freq = std::stoi(data.find("vtk_out_freq")->second);
            params.vtk_output = (data.find("vtk_out_name_base")->second);
            params.x_min = convert(get_value(data, "x_min"));
            params.x_max = convert(get_value(data, "x_max"));
            params.y_min = convert(get_value(data, "y_min"));
            params.y_max = convert(get_value(data, "y_max"));
            params.z_min = convert(get_value(data, "z_min"));
            params.z_max = convert(get_value(data, "z_max"));
            params.x_n = std::stoi(get_value(data, "x_n"));
            params.y_n = std::stoi(get_value(data, "y_n"));
            params.z_n = std::stoi(get_value(data, "z_n"));
            params.r_cut = convert(get_value(data, "r_cut"));
            params.x_len = (params.x_max - params.x_min)/params.x_n;
            params.y_len = (params.y_max - params.y_min)/params.y_n;
            params.z_len = (params.z_max - params.z_min)/params.z_n;
            params.nl_vel = convert(get_value(data, "nl_vel"));
            params.nl_size = std::stoi(get_value(data, "nl_size"));
            params.nl_freq = std::stoi(get_value(data, "nl_freq"));
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
