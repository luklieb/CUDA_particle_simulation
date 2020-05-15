#pragma once
#include "classes.h"
#include "helper.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>


void update_list(const Params &params, std::vector<Particle> &particles, std::vector<int> &linked_cells, std::vector<int> &linked_particles);

void update_pos(const Params &params, std::vector<Particle> &particles);

void calc_velocity(const Params &params, std::vector<Particle> &particles);

void calc_force(const Params &params, std::vector<Particle> &particles, const std::vector<int> &linked_cells, const std::vector<int> &linked_particles);

//#### Hybrid methods ####

void update_list(const Params &params, std::vector<Particle> &particles, std::vector<int> &linked_cells, std::vector<int> &linked_particles, const int cell_border, std::vector<int> &active_particles, int &cntr);

void filter_particles(const Params &params, std::vector<Particle> &particles, const int cell_border, std::vector<Particle> &filtered_particles, int &cntr, const bool only_border);

void update_pos(const Params &params, std::vector<Particle> &particles, std::vector<int> &active_particles, const int cntr);

void calc_velocity(const Params &params, std::vector<Particle> &particles, std::vector<int> &active_particles, const int cntr);

void calc_force(const Params &params, std::vector<Particle> &particles, const std::vector<int> &linked_cells, const std::vector<int> &linked_particles, std::vector<int> &active_particles, const int cntr);

void replace_particles(std::vector<Particle> &particles, const std::vector<Particle> &filtered_particles, const int cntr);
