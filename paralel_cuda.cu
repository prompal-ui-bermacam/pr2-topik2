/**
*	Author : Rushikesh Gaidhani
*	Topic  : Matrix Multiplication on GPGPU using CUDA
*/

#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h> 

cudaEvent_t start, stop;  // using cuda events to measure time
float elapsed_time_ms;    // which is applicable for asynchronous code also

//Matrix multiplication kernel - thread specification
__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width)
{
    //2D Thread ID
    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y;
    
    //Pvalue stores the Pd element that is computed by the thread
    float Pvalue = 0;

    for(int k = 0; k < Width ; ++k){
        float Mdelement = Md[ty*Width + k];
        float Ndelement = Nd[k*Width + tx];
        Pvalue += (Mdelement*Ndelement);
    }
    Pd[ty*Width + tx] = Pvalue;
}

void MatrixMultiplication(float *M, float *N, float *P, int Width) 
{
    int size = Width*Width*sizeof(float);
    float *Md, *Nd, *Pd;
	int k = 100;
	int l = 100;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    //Transfer M and N to device memory
    cudaMalloc((void**)&Md, size);
    cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Nd, size);
    cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);
    
    //Allocate P on the device
    cudaMalloc((void**)&Pd,size);

    //Setup the execution configuration
    dim3 dimBlock((k-1)/Width+1,(l-1)/Width+1);
    dim3 dimGrid(Width,Width);

	cudaEventRecord(start, 0);			// use same timing*

    //Launch the device computation threads!
    MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,Width);

	//Transfer P from device to host
	cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     		// measure end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

    //Free device matrices
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}

int main(int argc, char** argv) 
{

    void MatrixMultiplication(float *, float *, float *, int);

	const int Width = atoi(argv[1]);
    float M[Width*Width], N[Width*Width], P[Width*Width];

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            M[i * Width + j] = i;
            N[i * Width + j] = j;
            P[i * Width + j] = 0;
        }
    }

    MatrixMultiplication(M, N, P, Width);
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%f \t", P[i * Width + j]);
        }
        printf("\n")
    }

	printf("Computation time of GPU: %f ms.\n This is a change", elapsed_time_ms);  // exe. time

    return 0;
}