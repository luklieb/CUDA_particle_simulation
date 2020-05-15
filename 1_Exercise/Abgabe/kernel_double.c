#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void colorJulia(global unsigned char *picture, unsigned int width, unsigned int height) {
    const size_t threshold = 10;
    const size_t max_iter = 200;
    size_t iter_cntr = 0;
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    double scaling = (double) width / 4;
    double a = (x / scaling - 2);
    double b = (y / scaling - 2);
    double tmp_a;



    while ( sqrt(a * a + b * b) < threshold ) {
        tmp_a = a * a - b * b - 0.8;
        b = a * b + b * a + 0.2;
        a = tmp_a;

        ++iter_cntr;
        if (iter_cntr >= max_iter) {
            break;
        }
    }
    if (x < width && y < height) {
        //picture[4 * width * y + 4 * x + 0] = (iter_cntr > 2) ? 255 : 0;
        picture[4 * width * y + 4 * x + 0] = iter_cntr;
        picture[4 * width * y + 4 * x + 1] = (iter_cntr > 25) ? iter_cntr-25 : 0;
        picture[4 * width * y + 4 * x + 2] = (iter_cntr > 10) ? iter_cntr-10 : 0;
        picture[4 * width * y + 4 * x + 3] = 255;
    }
}
