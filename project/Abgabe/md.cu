#include "reader_writer.h"
#include "cpu.h"
#include "gpu.h"
#include "helper.h"

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cout << "Usage: ./md_xxx [parameter file]" << std::endl;
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
    //TODO assert particles.length = nr_part

    // Create OutputWriter
    OutputWriter output_writer;

    // Init linked list for cell and particle parallel approach
    std::vector<int> linked_particles(params.num_part);
    std::vector<int> linked_cells(params.cells0 * params.cells1 * params.cells2, -1);

    // Data on device
    #if defined(GPU)
    Particle *d_particles;
    Params *d_params;
    int *d_linked_cells;
    int *d_linked_particles;

    const long long nBytes = sizeof(Particle) * (params.num_part);
    checkError(cudaMalloc(&d_particles, nBytes), "malloc particles");
    checkError(cudaMalloc(&d_params, sizeof(Params)), "malloc params");
    checkError(cudaMalloc(&d_linked_cells, sizeof(int) * params.cells0 * params.cells1 * params.cells2), "malloc linked cells");
    checkError(cudaMalloc(&d_linked_particles, sizeof(int) * params.num_part), "malloc linked particles");

    checkError(cudaMemcpy(d_particles, &particles[0], nBytes, cudaMemcpyHostToDevice), "memcpy host to device part");
    checkError(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice), "memcpy host to deviceparams");
    //TODO why?
    checkError(cudaMemcpy(d_linked_cells, &linked_cells[0], sizeof(int) * params.cells0 * params.cells1 * params.cells2, cudaMemcpyHostToDevice),
               "memcpy host to device cells");
    //TODO why?
    checkError(cudaMemcpy(d_linked_particles, &linked_particles[0], sizeof(int) * params.num_part, cudaMemcpyHostToDevice),
               "memcpy host to device linked particles");

    const dim3 threadsPerBlock(params.block_size);
    const dim3 numBlocks(params.num_part / params.block_size + 1);

    const dim3 numBlocksCells((params.cells0 * params.cells1 * params.cells2) / params.block_size + 1);

    #endif

    // Variables for measurement
    double total_time = 0.;
    double start_time, end_time;
    // Variables for iteration
    double time = 0.;
    size_t iter = 0, iter_v = 0;


    #if defined(GPU)
    update_list<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles);
    //checkError(cudaPeekAtLastError(), "peek error");
    //checkError(cudaDeviceSynchronize(), "");
    #else
    update_list(params, particles, linked_cells, linked_particles);
    #endif

    // Initial force calc.
    #if defined(GPU)
    calc_force<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles);
    //checkError(cudaDeviceSynchronize(), "");
    #else
    calc_force(params, particles, linked_cells, linked_particles);
    #endif

    while (time <= params.time_end) {
        if (iter % params.vtk_out_freq == 0) {
            #if defined(GPU)
            checkError(cudaMemcpy(&particles[0], d_particles, nBytes, cudaMemcpyDeviceToHost), "memcpy device to host vtk");
            #endif
            output_writer.write_vtk(particles, params, iter_v);
            ++iter_v;
        }


        start_time = getSeconds();
        #if defined(GPU)
        update_pos<<<numBlocks, threadsPerBlock>>>(d_params, d_particles);
        #else
        update_pos(params, particles);
        #endif

        #if defined(GPU)
        //checkError(cudaMemset(d_linked_cells, -1, sizeof(int) * params.cells0 * params.cells1 * params.cells2), "memset");
        //TODO numblocks etc
        set_list<<<numBlocksCells, threadsPerBlock>>>(d_linked_cells, params.cells0 * params.cells1 * params.cells2, -1);
        update_list<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles);
        calc_force<<<numBlocks, threadsPerBlock>>>(d_params, d_particles, d_linked_cells, d_linked_particles);
        #else
        linked_cells.assign(linked_cells.size(), -1);
        update_list(params, particles, linked_cells, linked_particles);
        calc_force(params, particles, linked_cells, linked_particles);
        #endif


        #if defined(GPU)
        calc_velocity<<<numBlocks, threadsPerBlock>>>(d_params, d_particles); 
        #else
        calc_velocity(params, particles);
        #endif

        checkError(cudaDeviceSynchronize(), "sync");
        end_time = getSeconds();
        total_time += end_time - start_time;
        if (iter % 100 == 0 && iter != 0) std:: cout << "time/iter: " << total_time/iter << std::endl;
        if (iter % 100 == 0 && iter != 0) std:: cout << "total: " << end_time - start_time << std::endl;

        time += params.timestep_length;
        ++iter;
    }

    // write last vtk file
    if (iter % params.vtk_out_freq == 0) {
        #if defined(GPU)
        checkError(cudaMemcpy(&particles[0], d_particles, nBytes, cudaMemcpyDeviceToHost), "memcpy device to host vtk");
        #endif
        output_writer.write_vtk(particles, params, iter_v);
    }

    std:: cout << total_time << std::endl;

    #if defined(GPU)
    checkError(cudaFree(d_params), "free");
    checkError(cudaFree(d_particles), "free");
    checkError(cudaFree(d_linked_cells), "free");
    checkError(cudaFree(d_linked_particles), "free");
    #endif

    exit(EXIT_SUCCESS);
}
