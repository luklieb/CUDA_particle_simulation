#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <sys/time.h>
#include "MDClasses.h"
#include "dem.h"


//######## Cell Indexingi ####################
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

//######## Linked List Stuff ####################
__global__ void update_list(const Params* params, Particle* particles, int* linked_cells, int* linked_particles) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < params->num_part){
        int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), params);
        linked_particles[id] = atomicExch(&(linked_cells[cell_index]), id);
    }
}

//######## Force Stuff ####################
    //id_a is body a, id_b is body b
    __device__ void add_collision_force(const Params* params, Particle* particles, int id_a, int id_b) {
        real r0, r1, r2;
        real v0, v1, v2;
        real x_len = params->x_max - params->x_min;
        real y_len = params->y_max - params->y_min;
        real z_len = params->z_max - params->z_min;
        real p; //penetration depth
        real dist; //distance of two bodies
        real v_n; //magnitude value of velocity in normal direction
        real f_n; //magnitude of force in normal direction

        r0 = particles[id_a].x0 - particles[id_b].x0;
        r1 = particles[id_a].x1 - particles[id_b].x1;
        r2 = particles[id_a].x2 - particles[id_b].x2;
        //minimum image criteria, depending on periodic or reflecting bc
        if(!params->reflect_x){
            if (r0 > x_len * 0.5) r0 = r0 - x_len;
            if (r0 <= -x_len * 0.5) r0 = r0 + x_len;
        }
        if(!params->reflect_y){
            if (r1 > y_len * 0.5) r1 = r1 - y_len;
            if (r1 <= -y_len * 0.5) r1 = r1 + y_len;
        }
        if(!params->reflect_z){
            if (r2 > z_len * 0.5) r2 = r2 - z_len;
            if (r2 <= -z_len * 0.5) r2 = r2 + z_len;
        }
        //calc norm (distance)
        dist = sqrt(pow(r0, 2.) + pow(r1, 2.) + pow(r2, 2.));
        //calc penetration depth
        p = particles[id_a].radius + particles[id_b].radius - dist;
        if(p >= 0){
            //calc normal
            r0 = r0/dist;
            r1 = r1/dist;
            r2 = r2/dist;
            //calc velocity difference
            v0 = particles[id_a].v0 - particles[id_b].v0;
            v1 = particles[id_a].v1 - particles[id_b].v1;
            v2 = particles[id_a].v2 - particles[id_b].v2;

            v_n = v0 * r0 + v1 * r1 + v2 * r2;
            f_n = params->k_s * p - params->k_dn * v_n;

            particles[id_a].force0 += f_n * r0;
            particles[id_a].force1 += f_n * r1;
            particles[id_a].force2 += f_n * r2;
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
            int id_b = linked_cells[neighbors[cell]]; // index of first particle in linked list
            while(id_b != -1 ){
                if ( id_b != id) {
                   add_collision_force(params, particles, id, id_b);
                }
                id_b = linked_particles[id_b];
            }
        }
        // Reflecting BC
        if(params->reflect_x){
            real p1 = particles[id].radius - abs(particles[id].x0 - params->x_min);
            real p2 = particles[id].radius - abs(particles[id].x0 - params->x_max);
            if(p1 >= 0){
                particles[id].force0 += params->k_s*p1 - params->k_dn*particles[id].v0; 
            }
            if(p2 >= 0){
                particles[id].force0 += -params->k_s*p2 - params->k_dn*particles[id].v0; 
            }
        }
        if(params->reflect_y){
            real p1 = particles[id].radius - abs(particles[id].x1 - params->y_min);
            real p2 = particles[id].radius - abs(particles[id].x1 - params->y_max);
            if(p1 >= 0){
                particles[id].force1 += params->k_s*p1 - params->k_dn*particles[id].v1; 
            }
            if(p2 >= 0){
                particles[id].force1 += -params->k_s*p2 - params->k_dn*particles[id].v1; 
            }
        }
        if(params->reflect_z){
            real p1 = particles[id].radius - abs(particles[id].x2 - params->z_min);
            real p2 = particles[id].radius - abs(particles[id].x2 - params->z_max);
            if(p1 >= 0){
                particles[id].force2 += params->k_s*p1 - params->k_dn*particles[id].v2; 
            }
            if(p2 >= 0){
                particles[id].force2 += -params->k_s*p2 - params->k_dn*particles[id].v2; 
            }
        }

        // Gravitation once
        if(particles[id].m != -1.){ 
            particles[id].force0 += params->g_x * particles[id].m;
            particles[id].force1 += params->g_y * particles[id].m;
            particles[id].force2 += params->g_z * particles[id].m;
        }
    }
}

//######## Main ####################
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
    std::vector<int> linked_particles(params.num_part);
    std::vector<int> linked_cells(params.x_n*params.y_n*params.z_n, -1);


    //Test output
    std::cout << "x_min: " << params.x_min << " x_max: " << params.x_max << " y_min: " << params.y_min << " y_max: " << params.y_max << " z_min: " << params.z_min << " z_max: " << params.z_max  << std::endl;

    // Data on device
    Particle* d_particles;
    Params* d_params;
    int* d_linked_cells;
    int* d_linked_particles;

    const long long nBytes = sizeof(Particle)*(params.num_part);
    checkError(cudaMalloc(&d_particles, nBytes), "malloc particles");
    checkError(cudaMalloc(&d_params, sizeof(Params)), "malloc params");
    checkError(cudaMalloc(&d_linked_cells, sizeof(int)*params.x_n*params.y_n*params.z_n), "malloc linked cells");
    checkError(cudaMalloc(&d_linked_particles, sizeof(int)*params.num_part), "malloc linked particles");


    checkError( cudaMemcpy(d_particles, &particles[0], nBytes, cudaMemcpyHostToDevice), "memcpy host to device part" );
    checkError( cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice), "memcpy host to deviceparams" );
    checkError( cudaMemcpy(d_linked_cells, &linked_cells[0], sizeof(int)*params.x_n*params.y_n*params.z_n, cudaMemcpyHostToDevice), "memcpy host to device cells" );
    checkError( cudaMemcpy(d_linked_particles, &linked_particles[0], sizeof(int)*params.num_part, cudaMemcpyHostToDevice), "memcpy host to device linked particles" );

    const dim3 threadsPerBlock(params.block_size);
    const dim3 numBlocks(params.num_part/params.block_size +1);

    // Variables for measurement
    double start_time, end_time, total_time = 0;
    // Variables for iteration
    double time = 0;
    size_t iter = 0, iter_p = 0, iter_v = 0;

    update_list <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
    checkError(cudaDeviceSynchronize(), "");
    //std::cout << 1 << std::endl;
    
    // Initial force calc.
    calc_force_pp <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
    checkError(cudaDeviceSynchronize(), "");
    //std::cout << 2 << std::endl;

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
            std::cout << "time: " << iter_v << std::endl;
        }
        
        calc_pos <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
        checkError(cudaDeviceSynchronize(), "");
        //std::cout << 3 << std::endl;

        start_time = getSeconds();
        
        checkError(cudaMemset(d_linked_cells, -1, sizeof(int)*params.x_n*params.y_n*params.z_n), "memset");
        checkError(cudaDeviceSynchronize(), "");
        update_list <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);
        checkError(cudaDeviceSynchronize(), "");
        //std::cout << 4 << std::endl;

        // Force calculation
        calc_force_pp <<< numBlocks, threadsPerBlock >>>(d_params, d_particles, d_linked_cells, d_linked_particles);        
        checkError(cudaDeviceSynchronize(), "");
        //std::cout << 5 << std::endl;

        end_time = getSeconds();
        total_time += end_time - start_time;

        calc_velocity <<< numBlocks, threadsPerBlock >>>(d_params, d_particles);
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

    std::cout << "total_time (force calc): " << total_time << std::endl;

    checkError(cudaFree(d_params), "free" );
    checkError(cudaFree(d_particles), "free" );
    checkError(cudaFree(d_linked_cells), "free" );
    checkError(cudaFree(d_linked_particles), "free" );

    exit( EXIT_SUCCESS );
}
