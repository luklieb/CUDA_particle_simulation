#pragma once
#include "classes.h"

__global__ void update_list(const Params *params, Particle *particles, int *linked_cells, int *linked_particles);

__global__ void update_pos(const Params *params, Particle *particles);

__global__ void calc_velocity(const Params *params, Particle *particles);

__global__ void calc_force(const Params *params, Particle *particles, const int *linked_cells, const int *linked_particles);

__global__ void set_list(int *list, int length, int value);

//#### Hybrid methods ####

__global__ void update_list(const Params *params, Particle *particles, int *linked_cells, int *linked_particles, const int cell_border, int *active_particles, int *cntr);

__global__ void filter_particles(const Params *params, Particle *particles, const int cell_border, Particle *filtered_particles, int *cntr, const bool only_border);

__global__ void update_pos(const Params *params, Particle *particles, const int *active_particles);

__global__ void calc_velocity(const Params *params, Particle *particles, const int *active_particles);

__global__ void calc_force(const Params *params, Particle *particles, const int *linked_cells, const int *linked_particles, const int *active_particles);

__global__ void replace_particles(Particle *particles, Particle *filtered_particles, const int cntr);
