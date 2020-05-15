#pragma once
#include <string>

//typedef double real;
typedef float real;

struct Params {
    std::string input;
    int block_size;
    int block_size_0;
    int block_size_1;
    int block_size_2;
    real timestep_length;
    real time_end;
    int part_out_freq;
    std::string part_output;
    int vtk_out_freq;
    std::string vtk_output;
    int num_part;
    real x0_min;
    real x0_max;
    real x1_min;
    real x1_max;
    real x2_max;
    real x2_min;
    int cells0; // anzahl zellen in x Richtung
    int cells1;
    int cells2;
    real length0; // zell laenge in x Richtung
    real length1;
    real length2;
    real r_cut;
    real epsilon;
    real sigma;
};

struct Particle {
    int id;
    bool on_gpu;
    real m;
    real radius;
    real x0;
    real x1;
    real x2;
    real v0;
    real v1;
    real v2;
    real force0;
    real force1;
    real force2;
    real force0_old;
    real force1_old;
    real force2_old;
};
