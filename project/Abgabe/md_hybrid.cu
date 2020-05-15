#include "reader_writer.h"
#include "cpu.h"
#include "gpu.h"
#include "helper.h"
#include <assert.h>

#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main(int argc, const char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./md_hybrid [parameter file] [0-1] [bool]" << std::endl;
        std::cerr << "The second paramter specifies the initial percentage of domain assigned to cpu." << std::endl;
        std::cerr << "The third paramter specifies if dynamic load balancing is used." << std::endl;
        exit(EXIT_FAILURE);
    }
    bool dynamic_lb;
    bool update_border;
    if (std::string(argv[3]) == "true") {
        dynamic_lb = true;
    } else if (std::string(argv[3]) == "false") {
        dynamic_lb = false;
    } else {
        std::cerr << "Wrong argument" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<Particle> particles;
    Params params;

    // Read parameter file and retrieve data
    ParameterReader params_reader;
    params_reader.read(std::string(argv[1]));
    params = params_reader.get();

    // Read input data and set num_part
    ParticleReader part_reader;
    params.num_part = part_reader.read(params);
    particles = part_reader.get();
    assert(particles.size() == params.num_part);

    // Create OutputWriter
    OutputWriter output_writer;

    // Init linked list for cell and particle parallel approach
    std::vector<int> linked_particles(params.num_part);
    std::vector<int> linked_cells(params.cells0 * params.cells1 * params.cells2, -1);
    // Hybrid stuff
    std::vector<int> active_particles(params.num_part, -1);
    std::vector<Particle> filtered_particles(params.num_part);
    std::vector<Particle> tmp_filtered_particles(params.num_part);
    int cntr = 0;
    int tmp_cntr = 0;
    // 0.33 CPU 0.66 GPU
    //int cell_border = params.cells0 * 0.;
    double border_factor = std::stod(argv[2]);
    int cell_border = params.cells0 * border_factor;
    bool only_border = true;


    // Data on device
    Particle *d_particles;
    Params *d_params;
    int *d_linked_cells;
    int *d_linked_particles;
    // Hybrid stuff
    int *d_active_particles;
    Particle *d_filtered_particles;
    int* d_cntr;

    const long long nBytes = sizeof(Particle) * (params.num_part);
    checkError(cudaMalloc(&d_particles, nBytes));
    checkError(cudaMalloc(&d_params, sizeof(Params)));
    checkError(cudaMalloc(&d_linked_cells, sizeof(int) * params.cells0 * params.cells1 * params.cells2));
    checkError(cudaMalloc(&d_linked_particles, sizeof(int) * params.num_part));
    checkError(cudaMalloc(&d_active_particles, sizeof(int) * params.num_part));
    checkError(cudaMalloc(&d_filtered_particles, nBytes));
    checkError(cudaMalloc(&d_cntr, sizeof(int)));

    checkError(cudaMemcpy(d_particles, &particles[0], nBytes, cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_linked_cells, &linked_cells[0], sizeof(int) * params.cells0 * params.cells1 * params.cells2, cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_linked_particles, &linked_particles[0], sizeof(int) * params.num_part, cudaMemcpyHostToDevice));

    checkError(cudaMemset(d_cntr, 0, sizeof(int)));

    const dim3 threadsPerBlock(params.block_size);
    const dim3 numBlocks(params.num_part / params.block_size + 1);
    const dim3 numBlocksCells((params.cells0 * params.cells1 * params.cells2) / params.block_size + 1);

    // Variables for measurement
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    double total_start, total_end, exch_start, exch_end, cpu_start, cpu_end;
    double total_time = 0.;
    double cpu_time = 0., gpu_time = 0.;
    // Variables for iteration
    double time = 0.;
    size_t iter = 0, iter_v = 0;

    //TODO

    set_list<<<numBlocks, threadsPerBlock>>>(d_active_particles, params.num_part, -1);
    update_list<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles, cell_border, d_active_particles, d_cntr);
            
    checkError(cudaMemcpy(&cntr, d_cntr, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "(initial) particles on gpu: " << cntr << std::endl;

    cntr = 0;
    update_list(params, particles, linked_cells, linked_particles, cell_border, active_particles, cntr);

    std::cout << "(initial) particles on cpu: " << cntr << std::endl;

    // Initial force calc.
    calc_force<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles, d_active_particles);
    calc_force(params, particles, linked_cells, linked_particles, active_particles, cntr);


    while (time <= params.time_end) {
        if (iter % params.vtk_out_freq == 0) {
            only_border = false;
            checkError(cudaMemset(d_cntr, 0, sizeof(int)));
            filter_particles<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, cell_border, d_filtered_particles, d_cntr, only_border);
            checkError(cudaMemcpy(&tmp_cntr, d_cntr, sizeof(int), cudaMemcpyDeviceToHost));
            checkError(cudaMemcpy(&filtered_particles[0], d_filtered_particles, sizeof(Particle)*tmp_cntr, cudaMemcpyDeviceToHost));
            replace_particles(particles, filtered_particles, tmp_cntr);
            output_writer.write_vtk(particles, params, iter_v);
            ++iter_v;
        }

        total_start = getSeconds();
        update_pos<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_active_particles);
        update_pos(params, particles, active_particles, cntr);

        //Exchange
        exch_start = getSeconds();
        /// When changing border, all particles have to be synchronized
        update_border = (iter % 100 == 0 && iter != 0);
        if ( dynamic_lb && update_border ) {
            only_border = false;
        } else {
            only_border = true;
        }
        checkError(cudaMemset(d_cntr, 0, sizeof(int)));
        filter_particles<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, cell_border, d_filtered_particles, d_cntr, only_border);
        checkError(cudaMemcpy(&tmp_cntr, d_cntr, sizeof(int), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(&tmp_filtered_particles[0], d_filtered_particles, sizeof(Particle)*tmp_cntr, cudaMemcpyDeviceToHost));
        cntr = 0;
        filter_particles(params, particles, cell_border, filtered_particles, cntr, only_border);
        checkError(cudaMemcpy(d_filtered_particles, &filtered_particles[0], sizeof(Particle)*cntr, cudaMemcpyHostToDevice));
        replace_particles(particles, tmp_filtered_particles, tmp_cntr);
        replace_particles<<<numBlocks, threadsPerBlock>>>(d_particles, d_filtered_particles, cntr);
        checkError(cudaDeviceSynchronize());
        exch_end = getSeconds();

        //Dynamic load balancing
        if ( dynamic_lb && update_border ) {
            //TODO case 0 or 1... error checking
            std::cout << "gpu: " << gpu_time << " cpu: " << cpu_time << std::endl;
            //only change if gpu/cpu not approx. equal
            if (std::abs(gpu_time/cpu_time - 1) > 0.1)
            border_factor = (gpu_time/cpu_time) * border_factor;
            if ((std::abs(gpu_time/cpu_time - 1) > 0.5) && ((border_factor < 0.01) || (border_factor > 0.99) ))
            border_factor = 0.5;
            std::cout << "time diff: " << std::abs(gpu_time/cpu_time - 1) << std::endl;
            cell_border =  border_factor * params.cells0;
            std::cout << "new border: " << border_factor << std::endl << std::endl;
            gpu_time = 0;
            cpu_time = 0;
        }

        cudaEventRecord(gpu_start);
        set_list<<<numBlocksCells, threadsPerBlock>>>(d_linked_cells, params.cells0 * params.cells1 * params.cells2, -1);
        set_list<<<numBlocks, threadsPerBlock>>>(d_active_particles, params.num_part, -1);
        checkError(cudaMemset(d_cntr, 0, sizeof(int)));
        update_list<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles, cell_border, d_active_particles, d_cntr);

        calc_force<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles, d_active_particles);
        calc_velocity<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_active_particles);
        cudaEventRecord(gpu_end);

        cpu_start = getSeconds();
        linked_cells.assign(linked_cells.size(), -1);
        active_particles.assign(active_particles.size(), -1);
        cntr = 0;
        update_list(params, particles, linked_cells, linked_particles, cell_border, active_particles, cntr);
        calc_force(params, particles, linked_cells, linked_particles, active_particles, cntr);
        calc_velocity(params, particles, active_particles, cntr);
        cpu_end = getSeconds();

        cudaEventSynchronize(gpu_end);
        float tmp_gpu_time = 0.;
        cudaEventElapsedTime(&tmp_gpu_time, gpu_start, gpu_end);
        gpu_time += tmp_gpu_time/1000;
        total_end = getSeconds();
        total_time += total_end - total_start;
        cpu_time += cpu_end - cpu_start;

        //TODO
        /*
        if (iter % 100 == 0 && iter != 0) std:: cout << "time/iter: " << total_time/iter << std::endl;
        if (iter % 100 == 0 && iter != 0) std:: cout << "total: " << total_end - total_start << std::endl;
        if (iter % 100 == 0 && iter != 0) std:: cout << "exch: " << exch_end - exch_start  << std::endl;
        if (iter % 100 == 0 && iter != 0) std:: cout << "cpu: " << cpu_end - cpu_start  << std::endl;
        if (iter % 100 == 0 && iter != 0) std:: cout << "gpu: " << tmp_gpu_time/1000  << std::endl;
        */

        time += params.timestep_length;
        ++iter;
    }

    checkError(cudaDeviceSynchronize());
    // write last vtk file
    if (iter % params.vtk_out_freq == 0) {
        only_border = false;
        checkError(cudaMemset(d_cntr, 0, sizeof(int)));
        filter_particles<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, cell_border, d_filtered_particles, d_cntr, only_border);
        checkError(cudaMemcpy(&cntr, d_cntr, sizeof(int), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(&filtered_particles[0], d_filtered_particles, sizeof(Particle)*cntr, cudaMemcpyDeviceToHost));
        replace_particles(particles, filtered_particles, cntr);
        output_writer.write_vtk(particles, params, iter_v);
    }

    std:: cout << total_time << std::endl;
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );
    std:: cout << "after" << std::endl;

    /*
    checkError(cudaFree(d_params));
    checkError(cudaFree(d_particles));
    checkError(cudaFree(d_linked_cells));
    checkError(cudaFree(d_linked_particles));
    checkError(cudaFree(d_active_particles));
    checkError(cudaFree(d_filtered_particles));
    checkError(cudaFree(d_cntr));
    */

    exit(EXIT_SUCCESS);
}
