#include "reader_writer.h"
//##### ParameterReader
void ParameterReader::read(const std::string &filename) {
    std::ifstream parameters(filename);
    std::string line;
    if (parameters.is_open()) {
        while (!parameters.eof()) {
            std::getline(parameters, line);
            if ((line.length() <= 1))
                continue;
            size_t pos = line.find_first_of(" ");
            size_t last = line.find_last_of(" ");
            //size_t ret = line.find_first_of("\n\r");
            data.insert(std::pair<std::string, std::string>(line.substr(0, pos), line.substr(last + 1, line.length())));
        }
    }
    parameters.close();
}

std::string ParameterReader::get_value(const std::string &key) {
    if (data.count(key) == 0) {
        std::cerr << "Key: " << key << ",does not exist." << std::endl;
        return "-1";
    } else {
        return data.at(key);
    }
}

Params ParameterReader::get() {
    Params params;
    params.input = get_value("part_input_file");
    params.block_size = std::stoi(get_value("cl_workgroup_1dsize"));
    params.block_size_0 = std::stoi(get_value("cl_workgroup_3dsize_0"));
    params.block_size_1 = std::stoi(get_value("cl_workgroup_3dsize_1"));
    params.block_size_2 = std::stoi(get_value("cl_workgroup_3dsize_2"));
    params.timestep_length = convert(get_value("timestep_length"));
    params.time_end = convert(get_value("time_end"));
    params.part_out_freq = std::stoi(get_value("part_out_freq"));
    params.part_output = get_value("part_out_name_base");
    params.vtk_out_freq = std::stoi(get_value("vtk_out_freq"));
    params.vtk_output = get_value("vtk_out_name_base");
    params.x0_min = convert(get_value("x0_min"));
    params.x0_max = convert(get_value("x0_max"));
    params.x1_min = convert(get_value("x1_min"));
    params.x1_max = convert(get_value("x1_max"));
    params.x2_min = convert(get_value("x2_min"));
    params.x2_max = convert(get_value("x2_max"));
    params.cells0 = std::stoi(get_value("cells0"));
    params.cells1 = std::stoi(get_value("cells1"));
    params.cells2 = std::stoi(get_value("cells2"));
    params.length0 = (params.x0_max - params.x0_min) / params.cells0;
    params.length1 = (params.x1_max - params.x1_min) / params.cells1;
    params.length2 = (params.x2_max - params.x2_min) / params.cells2;
    params.epsilon = convert(get_value("epsilon"));
    params.sigma = convert(get_value("sigma"));
    params.r_cut = convert(get_value("r_cut"));
    return params;
}

//##### ParticleReader
int ParticleReader::read(const Params &parameters) {
    std::ifstream input(parameters.input);
    std::string line;
    std::getline(input, line);
    int num_part = std::stoi(line);

    if (input.is_open()) {
        for (int i = 0; i < num_part; ++i) {
            Particle tmp;
            tmp.id = i;
            // Split line into 7 values and store them into the particle vector
            std::getline(input, line);
            int first = line.find_first_not_of(std::string(" "), 0);
            int second = line.find(" ", first);
            std::string mass = line.substr(first, second);
            if (mass == std::string("inf"))
                tmp.m = -1.;
            else
                tmp.m = convert(mass);
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first);
            tmp.radius = convert(line.substr(first, second));
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first);
            tmp.x0 = convert(line.substr(first, second));
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first);
            tmp.x1 = convert(line.substr(first, second));
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first);
            tmp.x2 = convert(line.substr(first, second));
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first);
            tmp.v0 = convert(line.substr(first, second));
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first);
            tmp.v1 = convert(line.substr(first, second));
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first);
            tmp.v2 = convert(line.substr(first, second));

            //TODO: not enough if x0 << x0_min etc.
            // Check if particles inside boundary
            if (tmp.x0 < parameters.x0_min) {
                tmp.x0 += parameters.x0_max - parameters.x0_min;
            } else if (tmp.x0 > parameters.x0_max) {
                tmp.x0 -= parameters.x0_max - parameters.x0_min;
            }
            if (tmp.x1 < parameters.x1_min) {
                tmp.x1 += parameters.x1_max - parameters.x1_min;
            } else if (tmp.x1 > parameters.x1_max) {
                tmp.x1 -= parameters.x1_max - parameters.x1_min;
            }
            if (tmp.x2 < parameters.x2_min) {
                tmp.x2 += parameters.x2_max - parameters.x2_min;
            } else if (tmp.x2 > parameters.x2_max) {
                tmp.x2 -= parameters.x2_max - parameters.x2_min;
            }

            particles.push_back(tmp);
        }
    }
    input.close();
    return num_part;
}

std::vector<Particle> ParticleReader::get() {
    return particles;
}

//##### OutputWriter
void OutputWriter::write_part(const std::vector<Particle> &particles, const Params &params, size_t iter_p) {
    std::ofstream result(params.part_output + std::to_string(iter_p) + ".out");
    result << particles.size() << std::endl;
    result << std::fixed;
    for (auto p : particles) {
        result << p.m << " " << p.x0 << " " << p.x1 << " " << p.x2 << " " << p.v0 << " " << p.v1 << " " << p.v2 << std::endl;
    }
    result.close();
}

void OutputWriter::write_vtk(const std::vector<Particle> &particles, const Params &params, size_t iter_v) {
    std::ofstream result(params.vtk_output + std::to_string(iter_v) + ".vtk");
    result << std::fixed;
    if (result.is_open()) {
        result << "# vtk DataFile Version 4.0" << std::endl;
        result << "hesp visualization file" << std::endl;
        result << "ASCII" << std::endl;
        result << "DATASET UNSTRUCTURED_GRID" << std::endl;
        result << "POINTS " << params.num_part << " double" << std::endl;
        for (auto p : particles)
            result << p.x0 << " " << p.x1 << " " << p.x2 << std::endl;

        result << "CELLS 0 0" << std::endl;
        result << "CELL_TYPES 0" << std::endl;
        result << "POINT_DATA " << params.num_part << std::endl;
        result << "SCALARS m double" << std::endl;
        result << "LOOKUP_TABLE default" << std::endl;
        for (auto p : particles)
            result << p.m << std::endl;

        result << "SCALARS r double" << std::endl;
        result << "LOOKUP_TABLE default" << std::endl;
        for (auto p : particles)
            result << p.radius << std::endl;

        result << "VECTORS v double" << std::endl;
        for (auto p : particles)
            result << p.v0 << " " << p.v1 << " " << p.v2 << std::endl;
        result << "SCALARS c double" << std::endl;
        result << "LOOKUP_TABLE default" << std::endl;
        for (auto p : particles)
            result << (int)p.on_gpu << std::endl;

   }
    result.close();
}
