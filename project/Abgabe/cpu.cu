#include "cpu.h"
#include "helper.h"
#include <assert.h>

void update_list(const Params &params, std::vector<Particle> &particles, std::vector<int> &linked_cells, std::vector<int> &linked_particles) {
    //#pragma omp parallel for num_threads(16)
    for (int id = 0; id < params.num_part; ++id) {
        int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), &params);
        #pragma omp critical
        {
            linked_particles[id] = linked_cells[cell_index];
            linked_cells[cell_index] = id;
        }
        particles[id].on_gpu = false;
    }
}

void update_pos(const Params &params, std::vector<Particle> &particles) {
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < params.num_part; ++id) {
        update_pos(id, &params, &particles[0]);
    }
}

void calc_velocity(const Params &params, std::vector<Particle> &particles) {
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < params.num_part; ++id) {
        calc_velocity(id, &params, &particles[0]);
    }
}

void calc_force(const Params &params, std::vector<Particle> &particles, const std::vector<int> &linked_cells, const std::vector<int> &linked_particles) {
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < params.num_part; ++id) {
        calc_force(id, &params, &particles[0], &linked_cells[0], &linked_particles[0]);
    }
}

//#### Hybrid methods ####

void update_list(const Params &params, std::vector<Particle> &particles, std::vector<int> &linked_cells, std::vector<int> &linked_particles, const int cell_border, std::vector<int> &active_particles, int &cntr) {
    assert(cntr == 0);
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < params.num_part; ++id) {
        //TODO: assert active_particles initalized with -1
        int cell_id0 = (particles[id].x0 - params.x0_min) / params.length0;
        if (cell_id0 < cell_border) {
            particles[id].on_gpu = false;
            int pos;
            #pragma omp atomic capture
            pos = cntr++;
            active_particles[pos] = id;
        } else {
            particles[id].on_gpu = true;
        }

        if (((cell_id0 <= cell_border) && (cell_id0 >= 0)) || (cell_id0 == params.cells0 - 1)) {
            int cell_index = calc_cell_index((particles[id].x0), (particles[id].x1), (particles[id].x2), &params);
            //TODO
            if (cell_index < 0) std::cout << "cellid0 " << cell_id0 << " part " << id << "x0 "<< particles[id].x0 << std::endl;
            #pragma omp critical
            {
                linked_particles[id] = linked_cells[cell_index];
                linked_cells[cell_index] = id;
            }
        }
    }
}

void filter_particles(const Params &params, std::vector<Particle> &particles, const int cell_border, std::vector<Particle> &filtered_particles, int &cntr, const bool only_border) {
    assert(cntr == 0);
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < params.num_part; ++id) {
        int cell_id0 = (particles[id].x0 - params.x0_min) / params.length0;
        // ALL: 0 to cells0 - 1
        // GPU: cell_border to cells0 - 1
        // CPU: 0 to cell_border - 1
        bool predicate = (only_border) ? !(particles[id].on_gpu) && !((cell_id0 < cell_border - 1) && (cell_id0 > 0)) : !(particles[id].on_gpu);
        if (predicate)  {
            int pos;
            #pragma omp atomic capture
            pos = cntr++;
            filtered_particles[pos] = particles[id];
        }
    }
}

void update_pos(const Params &params, std::vector<Particle> &particles, std::vector<int> &active_particles, const int cntr) {
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < cntr; ++id) {
        if(active_particles[id] != -1)
            update_pos(active_particles[id], &params, &particles[0]);
    }
}

void calc_velocity(const Params &params, std::vector<Particle> &particles, std::vector<int> &active_particles, const int cntr) {
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < cntr; ++id) {
        if(active_particles[id] != -1)
            calc_velocity(active_particles[id], &params, &particles[0]);
    }
}

void calc_force(const Params &params, std::vector<Particle> &particles, const std::vector<int> &linked_cells, const std::vector<int> &linked_particles, std::vector<int> &active_particles, const int cntr) {
    #pragma omp parallel for num_threads(16)
    for (int id = 0; id < cntr; ++id) {
        if(active_particles[id] != -1)
            calc_force(active_particles[id], &params, &particles[0], &linked_cells[0], &linked_particles[0]);
    }
}

void replace_particles(std::vector<Particle> &particles, const std::vector<Particle> &filtered_particles, const int cntr) {
    #pragma omp parallel for num_threads(16)
    for (int j = 0; j < cntr; ++j) {
        int id = filtered_particles[j].id;
        particles[id] = filtered_particles[j];
    }
}
