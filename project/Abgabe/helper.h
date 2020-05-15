#pragma once
#include "classes.h"
#include <cuda_runtime.h>
#include <sys/time.h>

double getSeconds();

void checkError(cudaError_t err, const char *msg);

__host__ __device__ void get_neighbors(const int index, int *neighbors, const Params *params);

__host__ __device__ int calc_cell_index(const real x, const real y, const real z, const Params *params);

__host__ __device__ void update_pos(const int &id, const Params *params, Particle *particles);

__host__ __device__ void calc_velocity(const int &id, const Params *params, Particle *particles);

__host__ __device__ void calc_force(const int id, const Params *params, Particle *particles, const int *linked_cells, const int *linked_particles);

__host__ __device__ void add_lj_force(const Params *params, Particle *particles, const int &id_a, const int &i);

__host__ __device__ real ljpotential(const real &n, const real& sigma, const real& epsilon);

//__host__ __device__ void add_collision_force(const Params *params, Particle *particles, const int &id_a, const int &id_b);
