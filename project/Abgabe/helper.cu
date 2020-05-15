#include "helper.h"
#include <iostream>
#include <cuda_runtime.h>
// TODO: cpp11
double getSeconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void checkError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " " << msg << std::endl;
        exit(-1);
    }
}

// Return an array of size 27 with indices of neighbors
// neighbors* must be of length 27 ints
__host__ __device__ void get_neighbors(const int index, int *neighbors, const Params *params) {
    int cell_id2 = index / (params->cells0 * params->cells1);
    int cell_id1 = (index - cell_id2 * (params->cells0 * params->cells1)) / params->cells0;
    int cell_id0 = index - cell_id2 * (params->cells0 * params->cells1) - cell_id1 * params->cells0;
    if (cell_id0 + cell_id1 * params->cells0 + cell_id2 * (params->cells0 * params->cells1) != index) printf("error indexing \n");
    int neighbour_id0, neighbour_id1, neighbour_id2;
    int counter = 0;

    for (int offset2 = -1; offset2 <= 1; ++offset2) {
        neighbour_id2 = cell_id2 + offset2;
        if (cell_id2 + offset2 < 0) {
            neighbour_id2 = params->cells2 - 1;
        } else if (cell_id2 + offset2 >= params->cells2) {
            neighbour_id2 = 0;
        }
        for (int offset1 = -1; offset1 <= 1; ++offset1) {
            neighbour_id1 = cell_id1 + offset1;
            if (cell_id1 + offset1 < 0) {
                neighbour_id1 = params->cells1 - 1;
            } else if (cell_id1 + offset1 >= params->cells1) {
                neighbour_id1 = 0;
            }
            for (int offset0 = -1; offset0 <= 1; ++offset0) {
                neighbour_id0 = cell_id0 + offset0;
                if (cell_id0 + offset0 < 0) {
                    neighbour_id0 = params->cells0 - 1;
                } else if (cell_id0 + offset0 >= params->cells0) {
                    neighbour_id0 = 0;
                }
                neighbors[counter++] = neighbour_id0 + neighbour_id1 * params->cells0 + neighbour_id2 * (params->cells0 * params->cells1);
            }
        }
    }
}

__host__ __device__ int calc_cell_index(const real x0, const real x1, const real x2, const Params *params) {
    int cell_id0 = (x0 - params->x0_min) / params->length0;
    int cell_id1 = (x1 - params->x1_min) / params->length1;
    int cell_id2 = (x2 - params->x2_min) / params->length2;
    if (cell_id2 < 0 || cell_id0 < 0 || cell_id1 < 0) printf("error in cell index calc\n");;
    return cell_id2 * (params->cells0 * params->cells1) + cell_id1 * (params->cells0) + cell_id0;
}


__host__ __device__ void update_pos(const int &id, const Params *params, Particle *particles) {
    real x, y, z;
    if (particles[id].m != -1.) {
        x = particles[id].x0 + params->timestep_length * particles[id].v0 +
            (particles[id].force0 * pow(params->timestep_length, 2.)) / (2. * particles[id].m);
        if (x < params->x0_min) {
            particles[id].x0 = x + (params->x0_max - params->x0_min);
        } else if (x > params->x0_max) {
            particles[id].x0 = x - (params->x0_max - params->x0_min);
        } else {
            particles[id].x0 = x;
        }

        y = particles[id].x1 + params->timestep_length * particles[id].v1 +
            (particles[id].force1 * pow(params->timestep_length, 2.)) / (2. * particles[id].m);
        if (y < params->x1_min) {
            particles[id].x1 = y + (params->x1_max - params->x1_min);
        } else if (y > params->x1_max) {
            particles[id].x1 = y - (params->x1_max - params->x1_min);
        } else {
            particles[id].x1 = y;
        }

        z = particles[id].x2 + params->timestep_length * particles[id].v2 +
            (particles[id].force2 * pow(params->timestep_length, 2.)) / (2. * particles[id].m);
        if (z < params->x2_min) {
            particles[id].x2 = z + (params->x2_max - params->x2_min);
        } else if (z > params->x2_max) {
            particles[id].x2 = z - (params->x2_max - params->x2_min);
        } else {
            particles[id].x2 = z;
        }
    }
    particles[id].force0_old = particles[id].force0;
    particles[id].force1_old = particles[id].force1;
    particles[id].force2_old = particles[id].force2;
}

__host__ __device__ void calc_velocity(const int &id, const Params *params, Particle *particles) {
    if (particles[id].m != -1.) {
        particles[id].v0 =
            particles[id].v0 + ((particles[id].force0_old + particles[id].force0) * params->timestep_length) / (2. * particles[id].m);
        particles[id].v1 =
            particles[id].v1 + ((particles[id].force1_old + particles[id].force1) * params->timestep_length) / (2. * particles[id].m);
        particles[id].v2 =
            particles[id].v2 + ((particles[id].force2_old + particles[id].force2) * params->timestep_length) / (2. * particles[id].m);
    }
}


__host__ __device__ void calc_force(const int id, const Params *params, Particle *particles, const int *linked_cells, const int *linked_particles) {
    int neighbors[27];
    particles[id].force0 = 0;
    particles[id].force1 = 0;
    particles[id].force2 = 0;

    // Get current cell index of this particle
    int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), params);
    // Get the 27 indices of the neighboring cells
    get_neighbors(cell_index, neighbors, params);

    // Loop over neighbour particles (in neighbour cells)
    for (int cell = 0; cell < 27; ++cell) {
        int id_b = linked_cells[neighbors[cell]]; // index of first particle in
                                                  // linked list
        while (id_b != -1) {
            if (id_b != id) {
                //add_collision_force(params, particles, id, id_b);
                add_lj_force(params, particles, id, id_b);
            }
            id_b = linked_particles[id_b];
        }
    }

    /*
    // Brute
    for (int id_b = 0; id_b < params->num_part; id_b++) {
        if (id_b != id) add_lj_force(params, particles, id, id_b);
    }
    */
}

__host__ __device__ real ljpotential(const real &n, const real& sigma, const real& epsilon) {
    return (24. * epsilon) / pow(n, 2.) * pow(sigma / n, 6.) * (2. * pow(sigma / n, 6.) - 1.);
}

__host__ __device__ void add_lj_force(const Params *params, Particle *particles, const int &id, const int &i) {
    real r0, r1, r2;
    real dist0 = params->x0_max - params->x0_min;
    real dist1 = params->x1_max - params->x1_min;
    real dist2 = params->x2_max - params->x2_min;
    real r_ij;
    real lenard;

    r0 = particles[id].x0 - particles[i].x0;
    r1 = particles[id].x1 - particles[i].x1;
    r2 = particles[id].x2 - particles[i].x2;

    // Minimum image criteria
    if (r0 > dist0 * 0.5) r0 = r0 - dist0;
    if (r1 > dist1 * 0.5) r1 = r1 - dist1;
    if (r2 > dist2 * 0.5) r2 = r2 - dist2;
    if (r0 <= -dist0 * 0.5) r0 = r0 + dist0;
    if (r1 <= -dist1 * 0.5) r1 = r1 + dist1;
    if (r2 <= -dist2 * 0.5) r2 = r2 + dist2;

    r_ij = sqrt(r0*r0 + r1*r1 + r2*r2);

    if(r_ij <= params->r_cut){
        lenard = ljpotential(r_ij, params->sigma, params->epsilon);
        particles[id].force0 += lenard*r0;
        particles[id].force1 += lenard*r1;
        particles[id].force2 += lenard*r2;
    }
}

