#include <iostream>
#include "lodepng.h"
#include <complex>
#include <vector>
#include <sys/time.h>

using::std::complex;


#define X_SIZE 2048
#define Y_SIZE 2048
#define ITER  200
#define RANGE  2.0
#define THRESHOLD  10.0

typedef float type;

void init(std::vector<complex<type>> & picture){

	int half1 = Y_SIZE >> 1;
	int half2 = X_SIZE >> 1;
	double normalized1 = 2.0*RANGE/Y_SIZE;
	double normalized2 = 2.0*RANGE/X_SIZE;
	for(int y = 0; y < half1; ++y){
		for(int x = 0; x < half2; ++x){
			//quadrant 0 (counterclockwise)
			picture[(half1+y)*X_SIZE + half2+x].imag(normalized1*y);
			picture[(half1+y)*X_SIZE + half2+x].real(normalized2*x);
			//quadrant 1
			picture[(half1+y)*X_SIZE + x].imag(normalized1*y);
			picture[(half1+y)*X_SIZE + x].real(normalized2*(x-half2));
			//quadrant 2
			picture[y*X_SIZE +x].imag(normalized1*(y-half1));
			picture[y*X_SIZE + x].real(normalized2*(x-half2));
			//quadrant 3
			picture[y*X_SIZE + half2+x].imag(normalized1*(y-half1));
			picture[y*X_SIZE + half2+x].real(normalized2*x);	

			//std::cout << picture[y*X_SIZE +x] << std::endl;
		}
	}
	
}

void update(std::vector<complex<type>> & picture, const complex<type> & c, std::vector<int> & colors){

	int counter = 0;
	for(int y = 0; y < Y_SIZE; ++y){
		for(int x = 0; x < X_SIZE; ++x){
			while(abs(picture[y*X_SIZE+x]) < THRESHOLD){
				picture[y*X_SIZE + x] = picture[y*X_SIZE + x]*picture[y*X_SIZE + x] + c;
				//std::cout << std::abs(picture[y*X_SIZE+x]) << std::endl;
				++counter;
				if(counter >= ITER)
					break;
			}
			colors[y*X_SIZE+x] = counter;
			counter = 0;
		}
	}
}


void color(std::vector<int> & colors){

	std::vector<unsigned char> out (X_SIZE*Y_SIZE*4U);
	for(int y = 0; y < Y_SIZE; ++y){
		for(int x = 0; x < X_SIZE; ++x){
			out[4*X_SIZE*y + 4*x + 0] = colors[y*X_SIZE+x]%255;;
			out[4*X_SIZE*y + 4*x + 1] = 0;
			out[4*X_SIZE*y + 4*x + 2] = 0;
			out[4*X_SIZE*y + 4*x + 3] = 255;
		}
	}
	unsigned error = lodepng::encode("julia.png", out.data(), X_SIZE, Y_SIZE);

	if(error) std::cout << "error lodepng: " << error << std::endl;
}


inline double getSeconds(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6 );
}

int main(int argc, const char * argv[]) {

	std::vector<complex<type>> picture (X_SIZE*Y_SIZE);
	const complex<type> c (-0.8, 0.2);
	std::vector<int> colors (X_SIZE*Y_SIZE, ITER);
	double start_time;
	double end_time;

	start_time = getSeconds();
	init(picture);
	update(picture, c, colors);
	color(colors);
	end_time = getSeconds();
	
	//for(auto i : colors)
	//	std::cout << i << std::endl;

	std::cout << end_time - start_time << std::endl;

	return 0;
}

