void checkError( cudaError_t err, const char* msg )
{
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " " << msg << std::endl;
        exit(-1);
    }
}

inline double getSeconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6 );
}

// Reads the file in filename and fills the particle array, it also returns the number of paricles in nr_particles
void read_particle_data(Params & parameters, std::vector<Particle> & particles){
    std::ifstream input(parameters.input);
    std::string line;
    std::getline(input, line);
    int num_particles = std::stoi(line);
    parameters.num_part = num_particles;

    if (input.is_open()){
        for(int i=0; i<num_particles; ++i){
            Particle tmp;
            // Split line into 7 values and store them into the particle vector
            std::getline(input, line);
            int first = line.find_first_not_of(std::string(" "), 0);
            int second = line.find(" ", first );
            tmp.m = stod( line.substr(first, second) );
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first );
            tmp.x0 = stod( line.substr(first, second) );
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first );
            tmp.x1 = stod( line.substr(first, second) );
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first );
            tmp.x2 = stod( line.substr(first, second) );
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first );
            tmp.v0 = stod( line.substr(first, second) );
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first );
            tmp.v1 = stod( line.substr(first, second) );
            first = line.find_first_not_of(std::string(" "), second);
            second = line.find(" ", first );
            tmp.v2 = stod( line.substr(first, second) );

            //check ob particle in den boundaries liegen
            if(tmp.x0 < parameters.x_min){
                tmp.x0 += parameters.x_max  - parameters.x_min;
            }else if(tmp.x0 > parameters.x_max){
                tmp.x0 -= parameters.x_max - parameters.x_min;
            }
            if(tmp.x1 < parameters.y_min){
                tmp.x1 += parameters.y_max - parameters.y_min;
            }else if(tmp.x1 > parameters.y_max){
                tmp.x1 -= parameters.y_max - parameters.y_min;
            }
            if(tmp.x2 < parameters.z_min){
                tmp.x2 += parameters.z_max - parameters.z_min;
            }else if(tmp.x2 > parameters.z_max){
                tmp.x2 -= parameters.z_max - parameters.z_min;
            }

            particles.push_back(tmp);
        }
    }
    input.close();
}

__global__ void calc_pos(const Params* params, Particle* particles) { 
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    real x, y, z;

    if(id < params->num_part){

        //periodische Randbedingungen fur x Koordinate
        x = particles[id].x0 + params->timestep_length * particles[id].v0 + particles[id].force0 * pow(params->timestep_length, 2.) / (2. * particles[id].m);
        if(x < params->x_min){
            particles[id].x0 = x + (params->x_max - params->x_min);
        }else if(x > params->x_max){
            particles[id].x0 = x - (params->x_max - params->x_min);
        }else{
            particles[id].x0 = x;
        }

        y = particles[id].x1 + params->timestep_length * particles[id].v1 + particles[id].force1 * pow(params->timestep_length, 2.) / (2. * particles[id].m);
        if(y < params->y_min){
            particles[id].x1 = y + (params->y_max - params->y_min);
        }else if(y > params->y_max){
            particles[id].x1 = y - (params->y_max - params->y_min);
        }else{
            particles[id].x1 = y;
        }

        z = particles[id].x2 + params->timestep_length * particles[id].v2 + particles[id].force2 * pow(params->timestep_length, 2.) / (2. * particles[id].m);
        if(z < params->z_min){
            particles[id].x2 = z + (params->z_max - params->z_min);
        }else if(z > params->z_max){
            particles[id].x2 = z - (params->z_max - params->z_min);
        }else{
            particles[id].x2 = z;
        }

        particles[id].force0_old = particles[id].force0;
        particles[id].force1_old = particles[id].force1;
        particles[id].force2_old = particles[id].force2;
    }
}

__global__ void calc_velocity(const Params* params, Particle* particles) { 
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < params->num_part){
        particles[id].v0 = particles[id].v0 + (particles[id].force0_old + particles[id].force0) * params->timestep_length / (2. * particles[id].m);
        particles[id].v1 = particles[id].v1 + (particles[id].force1_old + particles[id].force1) * params->timestep_length / (2. * particles[id].m);
        particles[id].v2 = particles[id].v2 + (particles[id].force2_old + particles[id].force2) * params->timestep_length / (2. * particles[id].m);
    }
}

inline void write_part_output(std::vector<Particle>& particles, Params& params, size_t iter_p) {
    std::ofstream result(params.part_output + std::to_string(iter_p) + ".out");
    result << particles.size() << std::endl;
    result << std::fixed;
    for (auto p : particles) {
        result << p.m << " " << p.x0 << " " << p.x1 << " " << p.x2 << " " << p.v0 << " " << p.v1 << " " << p.v2 << std::endl;
    }
    result.close();
}

inline void write_vtk_output(const std::vector<Particle>& particles, const Params& params, size_t iter_v) {
    std::ofstream result(params.vtk_output + std::to_string(iter_v) + ".vtk");
    result << std::fixed;
    if(result.is_open()){
        result << "# vtk DataFile Version 4.0" << std::endl;
        result << "hesp visualization file" << std::endl;
        result << "ASCII" << std::endl;
        result << "DATASET UNSTRUCTURED_GRID" << std::endl;
        result << "POINTS " << params.num_part << " double" << std::endl;
        for(auto p : particles)
            result << p.x0 << " " << p.x1 << " " << p.x2 << std::endl;

        result << "CELLS 0 0" << std::endl;
        result << "CELL_TYPES 0" << std::endl;
        result << "POINT_DATA " << params.num_part << std::endl;
        result << "SCALARS m double" << std::endl;
        result << "LOOKUP_TABLE default" << std::endl;
        for (auto p : particles)
            result << p.m << std::endl;

        result << "VECTORS v double" << std::endl;
        for(auto p : particles)
            result << p.v0 << " " << p.v1 << " " << p.v2 << std::endl;
    }
    result.close();
}
