#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <typeinfo>
#include <vector>
#include <sys/time.h>
#include "MDClasses.h"

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

			particles.push_back(tmp);
		}
	}
	input.close();
}

__device__ real norm(const Particle& p1, const Particle& p2) {
	return sqrt(pow(p1.x0 - p2.x0, 2.) + pow(p1.x1 - p2.x1, 2.) + pow(p1.x2 - p2.x2, 2.));
}

__device__ real ljpotential(const Particle& p1, const Particle& p2, const real& sigma, const real& epsilon) {
	return (24. * epsilon) / pow(norm(p1, p2), 2.) * pow(sigma / norm(p1, p2), 6.) * ( 2 * pow(sigma / norm(p1, p2), 6.) - 1);
}

__global__ void calc_force(Params* params, Particle* particles) { 
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < params->num_part){
		particles[id].force0 = 0;
		particles[id].force1 = 0;
		particles[id].force2 = 0;
		real factor;
		for (int i = 0; i < params->num_part; ++i) {
			if ( i != id ) {
				factor = ljpotential(particles[id], particles[i], params->sigma, params->epsilon);
				particles[id].force0 += factor * (particles[id].x0 - particles[i].x0);
				particles[id].force1 += factor * (particles[id].x1 - particles[i].x1);
				particles[id].force2 += factor * (particles[id].x2 - particles[i].x2);
			}
		}
	}
}

__global__ void calc_pos(Params* params, Particle* particles) { 
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < params->num_part){
		particles[id].x0 = particles[id].x0 + params->timestep_length * particles[id].v0 + particles[id].force0 * pow(params->timestep_length, 2.) / (2. * particles[id].m);
		particles[id].x1 = particles[id].x1 + params->timestep_length * particles[id].v1 + particles[id].force1 * pow(params->timestep_length, 2.) / (2. * particles[id].m);
		particles[id].x2 = particles[id].x2 + params->timestep_length * particles[id].v2 + particles[id].force2 * pow(params->timestep_length, 2.) / (2. * particles[id].m);
		particles[id].force0_old = particles[id].force0;
		particles[id].force1_old = particles[id].force1;
		particles[id].force2_old = particles[id].force2;
	}
}

__global__ void calc_velocity(Params* params, Particle* particles) { 
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
	result << std::setprecision(6) << std::fixed;
	for (auto p : particles) {
		result << p.m << " " << p.x0 << " " << p.x1 << " " << p.x2 << " " << p.v0 << " " << p.v1 << " " << p.v2 << std::endl;
	}
	result.close();
}

inline void write_vtk_output(std::vector<Particle>& particles, Params& params, size_t iter_v) {
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

int main(int argc, const char** argv){
	if( argc != 2 ){
		std::cout << "Usage: ./hesp [parameter file]" << std::endl;
		exit( EXIT_FAILURE );
	}

	double start_time, end_time;

	std::vector<Particle> particles;
	Params params;

	// Read parameter file and retrieve data
	ParameterReader params_reader;
	params_reader.read(std::string(argv[1]));
	params = params_reader.get();

	// Read input data
	read_particle_data(params, particles);

	// Particles on device
	Particle* d_particles;
	Params* d_params;	

	const long long nBytes = sizeof(Particle)*(params.num_part);
	checkError(cudaMalloc(&d_particles, nBytes), "malloc particles");
	checkError(cudaMalloc(&d_params, sizeof(Params)), "malloc params");

	checkError( cudaMemcpy(d_particles, &particles[0], nBytes, cudaMemcpyHostToDevice), "memcpy host to device part" );
	checkError( cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice), "memcpy host to deviceparams" );

	const dim3 threadsPerBlock(params.block_size);
	const dim3 numBlocks(params.num_part/params.block_size +1);


	double time = 0;
	size_t iter = 0, iter_p = 0, iter_v = 0;
	
	start_time = getSeconds();
	// Initial force calc.
	calc_force <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
	while (time <= params.time_end) {		
		if (iter % params.part_out_freq == 0) {
			checkError(cudaMemcpy(&particles[0], d_particles, nBytes, cudaMemcpyDeviceToHost), "memcpy device to host part" );
			write_part_output(particles, params, iter_p);
			++iter_p;
		}
		if (iter % params.vtk_out_freq == 0) {
			checkError(cudaMemcpy(&particles[0], d_particles, nBytes, cudaMemcpyDeviceToHost), "memcpy device to host vtk" );
			write_vtk_output(particles, params, iter_v);
			++iter_v;
		}

		calc_pos <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
		calc_force <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
		calc_velocity <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);

		//TODO needed after each kernel call ?
		checkError(cudaPeekAtLastError(), "");
		checkError(cudaDeviceSynchronize(), "");

		time += params.timestep_length;
		++iter;
	}

	end_time = getSeconds();
	std::cout << end_time - start_time << std::endl;

	checkError(cudaFree(d_params), "free" );
	checkError(cudaFree(d_particles), "free" );


	exit( EXIT_SUCCESS );


}
