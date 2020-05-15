#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <sys/time.h>
#include "MDClasses.h"
#include "md_periodic.h"


//########Cell Indexing####################
//Return an array of size 27 with indices of neighbors
//neighbors* must be of length 27 ints
__device__ void get_neighbors(const int index, int* neighbors, const Params* params ){
    int zkod = index / (params->x_n * params->y_n);
    int ykod = (index - zkod*(params->x_n*params->y_n)) / params->x_n;
    int xkod = index - zkod*(params->x_n*params->y_n) - ykod*params->x_n;
    int xkod_up, ykod_up, zkod_up;
    int counter = 0;

    for(int z=-1; z<=1; ++z){
        zkod_up = zkod+z;
        if(zkod+z < 0){
            zkod_up = params->z_n - 1;
        }else if(zkod+z >= params->z_n){
            zkod_up = 0;
        }
        for(int y=-1; y<=1; ++y){
            ykod_up = ykod+y;
            if(ykod+y < 0){
                ykod_up = params->y_n - 1;
            }else if(ykod+y >= params->y_n){
                ykod_up = 0;
            }
            for(int x=-1; x<=1; ++x){
                xkod_up = xkod+x;
                if(xkod+x < 0){
                    xkod_up = params->x_n - 1;
                }else if(xkod+x >= params->x_n){
                    xkod_up = 0;
                }
                neighbors[counter++] = xkod_up + ykod_up*params->x_n + zkod_up*(params->x_n * params->y_n);
            }
        }
    }
}

__device__ inline int calc_cell_index(const real x, const real y, const real z, const Params * params){
    int x_i = (x - params->x_min)/params->x_len;
    int y_i = (y - params->y_min)/params->y_len;
    int z_i = (z - params->z_min)/params->z_len;
    return z_i*(params->x_n*params->y_n) + y_i*(params->x_n) + x_i;
}

__global__ void update_list(const Params* params, Particle* particles, int* linked_cells, int* linked_particles) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < params->num_part){
        int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), params);
        linked_particles[id] = atomicExch(&(linked_cells[cell_index]), id);
    }
}

//########Force Stuff####################
/*
__device__ real norm(const Particle& p1, const Particle& p2) {
    return sqrt(pow(p1.x0 - p2.x0, 2.) + pow(p1.x1 - p2.x1, 2.) + pow(p1.x2 - p2.x2, 2.));
}

__device__ real ljpotential(const Particle& p1, const Particle& p2, const real& sigma, const real& epsilon) {
    real n = norm(p1, p2);
    return (24. * epsilon) / pow(n, 2.) * pow(sigma / n, 6.) * ( 2. * pow(sigma / n, 6.) - 1.);
}

*/

__device__ real ljpotential(const real & n, const real& sigma, const real& epsilon) {
    return (24. * epsilon) / pow(n, 2.) * pow(sigma / n, 6.) * (2. * pow(sigma / n, 6.) - 1.);
}

__device__ void update_force(const Params* params, Particle* particles, int id, int i) {
    real r0, r1, r2;
    real x_len = params->x_max - params->x_min;
    real y_len = params->y_max - params->y_min;
    real z_len = params->z_max - params->z_min;
    real r_ij;
    real lenard;


    r0 = particles[id].x0 - particles[i].x0;
    r1 = particles[id].x1 - particles[i].x1;
    r2 = particles[id].x2 - particles[i].x2;

    //minimum image criteria
    if (r0 > x_len * 0.5) r0 = r0 - x_len;
    if (r1 > y_len * 0.5) r1 = r1 - y_len;
    if (r2 > z_len * 0.5) r2 = r2 - z_len;
    if (r0 <= -x_len * 0.5) r0 = r0 + x_len;
    if (r1 <= -y_len * 0.5) r1 = r1 + y_len;
    if (r2 <= -z_len * 0.5) r2 = r2 + z_len;

    r_ij = sqrt(r0*r0 + r1*r1 + r2*r2);

    if(r_ij <= params->r_cut){
        lenard = ljpotential(r_ij, params->sigma, params->epsilon);
        particles[id].force0 += lenard*r0;
        particles[id].force1 += lenard*r1;
        particles[id].force2 += lenard*r2;
    }
}

__global__ void calc_force_brute(const Params* params, Particle* particles) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part) {
        particles[id].force0 = 0;
        particles[id].force1 = 0;
        particles[id].force2 = 0;
        for (int i = 0; i < params->num_part; ++i) {
            if ( i != id) {
                update_force(params, particles, id, i);
            }
        }
    }
}

__global__ void calc_force_pp(const Params* params, Particle* particles, const int* linked_cells, const int* linked_particles) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    int neighbors [27];
    if (id < params->num_part) {
        particles[id].force0 = 0;
        particles[id].force1 = 0;
        particles[id].force2 = 0;

        // Get current cell index of this particle
        int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), params);
        // Get the 27 indices of the neighboring cells
        get_neighbors(cell_index, neighbors, params);

        // Loop over neighbour particles (in neighbour cells)
        for (int cell = 0; cell < 27; ++cell) {
            int i = linked_cells[neighbors[cell]]; // index of first particle in linked list
            while(i != -1 ){
                if ( i != id) {
                   update_force(params, particles, id, i);
                }
                i = linked_particles[i];
            }
        }
    }
}

__global__ void calc_force_cp(const Params* params, Particle* particles, const int* linked_cells, const int* linked_particles) {
    //3D indexing
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int id = block_id * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int neighbors [27];

    if (id < params->x_n * params->y_n * params->z_n) {
        int cell_index = id;
        get_neighbors(cell_index, neighbors, params);

        int i = linked_cells[cell_index]; // index of first particle in linked list

        // Loop over particles in current cell
        while(i != -1 ){
            particles[i].force0 = 0;
            particles[i].force1 = 0;
            particles[i].force2 = 0;
            // Loop over neighbour particles (in neighbour cells)
            for (int cell = 0; cell < 27; ++cell) {
                int j = linked_cells[neighbors[cell]]; // index of first particle in linked list
                while(j != -1 ){
                    if ( i != j) {
                        update_force(params, particles, i, j);
                    }
                    j = linked_particles[j];
                }
            }
            i = linked_particles[i];
        }
    }
}


__global__ void calc_neighbours(const Params* params, Particle* particles, int* neighbour_list) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int first = id*params->nl_size; //first index of neighbour list of particle id
    const int last = first + params->nl_size - 1; //last index of neighbour list of particle id
    int counter = first;
    real r_ext = params->r_cut + params->nl_vel * params->timestep_length * params->nl_freq;

    for (int i = 0; i < params->num_part; ++i) {
        if (counter > last) break;
        if ( i != id) {
            real r0, r1, r2;
            real x_len = params->x_max - params->x_min;
            real y_len = params->y_max - params->y_min;
            real z_len = params->z_max - params->z_min;
            real r_ij;

            r0 = particles[id].x0 - particles[i].x0;
            r1 = particles[id].x1 - particles[i].x1;
            r2 = particles[id].x2 - particles[i].x2;

            //minimum image criteria
            if (r0 > x_len * 0.5) r0 = r0 - x_len;
            if (r1 > y_len * 0.5) r1 = r1 - y_len;
            if (r2 > z_len * 0.5) r2 = r2 - z_len;
            if (r0 <= -x_len * 0.5) r0 = r0 + x_len;
            if (r1 <= -y_len * 0.5) r1 = r1 + y_len;
            if (r2 <= -z_len * 0.5) r2 = r2 + z_len;

            r_ij = sqrt(r0*r0 + r1*r1 + r2*r2);

            if(r_ij <= r_ext){
                neighbour_list[counter] = i;
            	counter++;
			}
        }
    }
}
__global__ void calc_force_nl(const Params* params, Particle* particles, const int* neighbour_list) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part) {
        const int first = id*params->nl_size; //first index of neighbour list of particle id
        const int last = first + params->nl_size - 1; //last index of neighbour list of particle id

        particles[id].force0 = 0;
        particles[id].force1 = 0;
        particles[id].force2 = 0;

        // Loop over neighbour particles (in neighbour list)
        for (int l = first; l <= last; ++l) {
            // Get particle id from neighbour list
            //printf("id: %d, l: %d\n", id, l);
			int i = neighbour_list[l];
            // -1 in neighbour list = no more neighbours
            if (i < 0 || i > params->num_part) break;
            if ( i != id) {
					//printf("update\n");
                   update_force(params, particles, id, i);
            }
        }
    }
}

int main(int argc, const char** argv){
    if( argc != 2 ){
        std::cout << "Usage: ./md_xxx [parameter file]" << std::endl;
        exit( EXIT_FAILURE );
    }


    std::vector<Particle> particles;
    Params params;

    // Read parameter file and retrieve data
    ParameterReader params_reader;
    params_reader.read(std::string(argv[1]));
    params = params_reader.get();

    // Read input data // also sets num_part
    read_particle_data(params, particles);


    // Init linked list for cell and particle parallel approach
    #if defined(CELLP) || defined(PARTICLEP)
    std::vector<int> linked_particles(params.num_part);
    std::vector<int> linked_cells(params.x_n*params.y_n*params.z_n, -1);
    #endif

    if(params.r_cut > params.x_len || params.r_cut > params.y_len || params.r_cut > params.z_len){
        std::cout << "r_cut zu klein" << std::endl;
        exit( EXIT_FAILURE );
    }

    //Test output
    //std::cout << "x_min: " << params.x_min << " x_max: " << params.x_max << " y_min: " << params.y_min << " y_max: " << params.y_max << " z_min: " << params.z_min << " z_max: " << params.z_max << " r_cut: " << params.r_cut << " epsilon: " << params.epsilon << std::endl;

    // Data on device
    Particle* d_particles;
    Params* d_params;
    #if defined(CELLP) || defined(PARTICLEP)
    int* d_linked_cells;
    int* d_linked_particles;
    #elif defined(LIST)
    int* d_neighbour_list;
    #endif

    const long long nBytes = sizeof(Particle)*(params.num_part);
    checkError(cudaMalloc(&d_particles, nBytes), "malloc particles");
    checkError(cudaMalloc(&d_params, sizeof(Params)), "malloc params");
    #if defined(CELLP) || defined(PARTICLEP)
    checkError(cudaMalloc(&d_linked_cells, sizeof(int)*params.x_n*params.y_n*params.z_n), "malloc linked cells");
    checkError(cudaMalloc(&d_linked_particles, sizeof(int)*params.num_part), "malloc linked particles");
    #elif defined(LIST)
    checkError(cudaMalloc(&d_neighbour_list, sizeof(int)*params.num_part*params.nl_size), "malloc neighbour list");
    checkError(cudaMemset(d_neighbour_list, -1, sizeof(int)*params.num_part*params.nl_size), "memset neighbour");
    #endif

    checkError( cudaMemcpy(d_particles, &particles[0], nBytes, cudaMemcpyHostToDevice), "memcpy host to device part" );
    checkError( cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice), "memcpy host to deviceparams" );
    #if defined(CELLP) || defined(PARTICLEP)
    checkError( cudaMemcpy(d_linked_cells, &linked_cells[0], sizeof(int)*params.x_n*params.y_n*params.z_n, cudaMemcpyHostToDevice), "memcpy host to device cells" );
    checkError( cudaMemcpy(d_linked_particles, &linked_particles[0], sizeof(int)*params.num_part, cudaMemcpyHostToDevice), "memcpy host to device linked particles" );
    #endif

    const dim3 threadsPerBlock(params.block_size);
    const dim3 numBlocks(params.num_part/params.block_size +1);

    const dim3 threadsPerBlock3D(params.block_size_x, params.block_size_y, params.block_size_z);
    const dim3 numBlocks3D(params.x_n/params.block_size_x + 1, params.y_n/params.block_size_y + 1, params.z_n/params.block_size_z + 1);

    // Variables for measurement
    double start_time, end_time, total_time = 0;
    // Variables for iteration
    double time = 0;
    size_t iter = 0, iter_p = 0, iter_v = 0;

    #if defined(CELLP) || defined(PARTICLEP)
    update_list <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
    #endif

    // Initial force calc.
    #if defined(BRUTE)
    calc_force_brute <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
    #elif defined(PARTICLEP)
    calc_force_pp <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
    #elif defined(CELLP)
    calc_force_cp <<< numBlocks3D, threadsPerBlock3D >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
    #elif defined(LIST)
    calc_neighbours <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_neighbour_list);
	checkError(cudaDeviceSynchronize(), "");
    calc_force_nl <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_neighbour_list);
	#endif
    checkError(cudaDeviceSynchronize(), "");

    while (time <= params.time_end) {
        if (iter % params.part_out_freq == 0) {
            checkError(cudaMemcpy(&particles[0], d_particles, nBytes, cudaMemcpyDeviceToHost), "memcpy device to host part");
            write_part_output(particles, params, iter_p);
            ++iter_p;
        }
        if (iter % params.vtk_out_freq == 0) {
            checkError(cudaMemcpy(&particles[0], d_particles, nBytes, cudaMemcpyDeviceToHost), "memcpy device to host vtk" );
            write_vtk_output(particles, params, iter_v);
            ++iter_v;
        }
		
        calc_pos <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
        
        start_time = getSeconds();
		
		#if defined(LIST)
        if (iter % params.nl_freq == 0) {
            checkError(cudaMemset(d_neighbour_list, -1, sizeof(int)*params.num_part*params.nl_size), "memset neighbour");
            calc_neighbours <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_neighbour_list);
            //printf("3\n");
			checkError(cudaDeviceSynchronize(), "");
        }
        #endif

        #if defined(CELLP) || defined(PARTICLEP)
        checkError(cudaMemset(d_linked_cells, -1, sizeof(int)*params.x_n*params.y_n*params.z_n), "memset");
        update_list <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
        #endif
        checkError(cudaDeviceSynchronize(), "");

        // Force calculation
        #if defined(BRUTE)
        calc_force_brute <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
        #elif defined(PARTICLEP)
        calc_force_pp <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
        #elif defined(CELLP)
        calc_force_cp <<< numBlocks3D, threadsPerBlock3D >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
        #elif defined(LIST)
        calc_force_nl <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_neighbour_list);
        //printf("4\n");
		#endif
        checkError(cudaDeviceSynchronize(), "");
        end_time = getSeconds();
        total_time += end_time - start_time;

        checkError(cudaDeviceSynchronize(), "");
        calc_velocity <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
        //printf("5\n");
		checkError(cudaPeekAtLastError(), "");
        checkError(cudaDeviceSynchronize(), "");

        time += params.timestep_length;
        ++iter;
    }

    //wird gebraucht fuer Szenarien mit der vtk_frequenz = 1 
    //--> letztes vtk file wird dann richtig ausgegeben
    if (iter % params.vtk_out_freq == 0) {
        checkError(cudaMemcpy(&particles[0], d_particles, nBytes, cudaMemcpyDeviceToHost), "memcpy device to host vtk" );
        write_vtk_output(particles, params, iter_v);
    }

    std::cout << params.num_part << " " << total_time << std::endl;

    checkError(cudaFree(d_params), "free" );
    checkError(cudaFree(d_particles), "free" );
    #if defined(CELLP) || defined(PARTICLEP)
    checkError(cudaFree(d_linked_cells), "free" );
    checkError(cudaFree(d_linked_particles), "free" );
    #elif defined(LIST)
    checkError(cudaFree(d_neighbour_list), "free" );
    #endif

    exit( EXIT_SUCCESS );
}
