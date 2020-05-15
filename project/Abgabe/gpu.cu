#include "gpu.h"
#include "helper.h"
#include <cuda_runtime.h>

__global__ void update_list(const Params *params, Particle *particles, int *linked_cells, int *linked_particles) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < params->num_part) {
        int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), params);
        linked_particles[id] = atomicExch(&(linked_cells[cell_index]), id);
        particles[id].on_gpu = true;
    }
}

__global__ void update_pos(const Params *params, Particle *particles) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part) update_pos(id, params, particles);
}

__global__ void calc_velocity(const Params *params, Particle *particles) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part) calc_velocity(id, params, particles);
}

__global__ void calc_force(const Params *params, Particle *particles, const int *linked_cells, const int *linked_particles) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part) calc_force(id, params, particles, linked_cells, linked_particles);
}

__global__ void set_list(int *list, int length, int value) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) list[id] = value;
}

//#### Hybrid methods ####

__global__ void update_list(const Params *params, Particle *particles, int *linked_cells, int *linked_particles, const int cell_border, int *active_particles, int *cntr) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    //TODO: assert active_particles initalized with -1
    if (id < params->num_part) {
        int cell_id0 = (particles[id].x0 - params->x0_min) / params->length0;
        if (cell_id0 >= cell_border) {
            particles[id].on_gpu = true;
            active_particles[atomicAdd(cntr, 1)] = id;
        } else {
            particles[id].on_gpu = false;
        }

        if (((cell_id0 >= cell_border - 1) && (cell_id0 <= params->cells0 - 1)) || (cell_id0 == 0)) {
            int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), params);
            //TODO
            if (cell_index < 0) printf("cellid0 %d part %d\n", cell_id0, id);
            linked_particles[id] = atomicExch(&(linked_cells[cell_index]), id);
        }
    }
}

__global__ void filter_particles(const Params *params, Particle *particles, const int cell_border, Particle *filtered_particles, int *cntr, const bool only_border) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    //TODO: assert filtered_particles initalized with ?? or correct size
    //TODO: assert cntr 0
    if (id < params->num_part) {
        int cell_id0 = (particles[id].x0 - params->x0_min) / params->length0;
        // ALL: 0 to cells0 - 1
        // GPU: cell_border to cells0 - 1
        // CPU: 0 to cell_border - 1
        bool predicate = (only_border) ? (particles[id].on_gpu) && !((cell_id0 > cell_border) && (cell_id0 < params->cells0 - 1)) : particles[id].on_gpu;
        if (predicate) {
            filtered_particles[atomicAdd(cntr, 1)] = particles[id];
        }
    }
}

__global__ void update_pos(const Params *params, Particle *particles, const int *active_particles) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part){
        id = active_particles[id];
        if (id != -1)
            update_pos(id, params, particles);
    }
}

__global__ void calc_velocity(const Params *params, Particle *particles, const int *active_particles) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part){
        id = active_particles[id];
        if (id != -1)
            calc_velocity(id, params, particles);
    }
}

__global__ void calc_force(const Params *params, Particle *particles, const int *linked_cells, const int *linked_particles, const int *active_particles) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < params->num_part){
        id = active_particles[id];
        if (id != -1)
            calc_force(id, params, particles, linked_cells, linked_particles);
    }
}

__global__ void replace_particles(Particle *particles, Particle *filtered_particles, const int cntr) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cntr){
        int id = filtered_particles[j].id;
        particles[id] = filtered_particles[j];
    }
}
