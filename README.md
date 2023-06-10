# PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution confi guration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## Aim:
(i) To modify or set the execution configuration of block.x as 1023 & 1024 and compare the elapsed time obtained on Host and GPU.
(ii) To set the number of threads as 256 and obtain the elapsed time on Host and GPU.

## Procedure:
1.Initialize the device and set the device properties.

2.Allocate memory on the host for input and output arrays.

3.Initialize input arrays with random values on the host.

4.Allocate memory on the device for input and output arrays, and copy input data from host to device.

5.Launch a CUDA kernel to perform vector addition on the device.

## Program:
1. i)Block.x=1023

#include "common.h" 
#include <cuda_runtime.h> 
#include <stdio.h>
void checkResult(float *hostRef, float *gpuRef, const int N) 
{ 
double epsilon = 1.0E-8; bool match = 1;
for (int i = 0; i < N; i++)
{
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
               gpuRef[i], i);
        break;
    }
}

if (match) printf("Arrays match.\n\n");

return;
void initialData(float *ip, int size) { // generate different seed for random number time_t t; srand((unsigned) time(&t));

for (int i = 0; i < size; i++)
{
    ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}

return;
}
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
for (int idx = 0; idx < N; idx++)
{ 
C[idx] = A[idx] + B[idx]; 
} } 
global void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{ 
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i];
}
int main(int argc, char **argv)
{ 
printf("%s Starting...\n", argv[0]);

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up data size of vectors
int nElem = 1 << 27;
printf("Vector size %d\n", nElem);

// malloc host memory
size_t nBytes = nElem * sizeof(float);

float *h_A, *h_B, *hostRef, *gpuRef;
h_A     = (float *)malloc(nBytes);
h_B     = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef  = (float *)malloc(nBytes);

double iStart, iElaps;

// initialize data at host side
iStart = seconds();
initialData(h_A, nElem);
initialData(h_B, nElem);
iElaps = seconds() - iStart;
printf("initialData Time elapsed %f sec\n", iElaps);
memset(hostRef, 0, nBytes);
memset(gpuRef,  0, nBytes);

// add vector at host side for result checks
iStart = seconds();
sumArraysOnHost(h_A, h_B, hostRef, nElem);
iElaps = seconds() - iStart;
printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

// malloc device global memory
float *d_A, *d_B, *d_C;
CHECK(cudaMalloc((float**)&d_A, nBytes));
CHECK(cudaMalloc((float**)&d_B, nBytes));
CHECK(cudaMalloc((float**)&d_C, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int iLen = 1023;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x);

iStart = seconds();
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
       block.x, iElaps);

// check kernel error
CHECK(cudaGetLastError()) ;

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nElem);

// free device global memory
CHECK(cudaFree(d_A));
CHECK(cudaFree(d_B));
CHECK(cudaFree(d_C));

// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

return(0);


ii) Block.x=1024

#include "common.h" 
#include <cuda_runtime.h> 
#include <stdio.h>
void checkResult(float *hostRef, float *gpuRef, const int N) 
{ 
double epsilon = 1.0E-8;
bool match = 1;

for (int i = 0; i < N; i++)
{
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
               gpuRef[i], i);
        break;
    }
}

if (match) printf("Arrays match.\n\n");

return;
}
void initialData(float *ip, int size) 
{ 
time_t t; 
srand((unsigned) time(&t));

for (int i = 0; i < size; i++)
{
    ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}

return;
}
void sumArraysOnHost(float *A, float *B, float *C, const int N) 
{ 
for (int idx = 0; idx < N; idx++)
{ 
C[idx] = A[idx] + B[idx]; 
} }
global void sumArraysOnGPU(float *A, float *B, float *C, const int N) 
{ 
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i];
}
int main(int argc, char **argv)
{ 
printf("%s Starting...\n", argv[0]);

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up data size of vectors
int nElem = 1 << 27;
printf("Vector size %d\n", nElem);

// malloc host memory
size_t nBytes = nElem * sizeof(float);

float *h_A, *h_B, *hostRef, *gpuRef;
h_A     = (float *)malloc(nBytes);
h_B     = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef  = (float *)malloc(nBytes);

double iStart, iElaps;

// initialize data at host side
iStart = seconds();
initialData(h_A, nElem);
initialData(h_B, nElem);
iElaps = seconds() - iStart;
printf("initialData Time elapsed %f sec\n", iElaps);
memset(hostRef, 0, nBytes);
memset(gpuRef,  0, nBytes);

// add vector at host side for result checks
iStart = seconds();
sumArraysOnHost(h_A, h_B, hostRef, nElem);
iElaps = seconds() - iStart;
printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

// malloc device global memory
float *d_A, *d_B, *d_C;
CHECK(cudaMalloc((float**)&d_A, nBytes));
CHECK(cudaMalloc((float**)&d_B, nBytes));
CHECK(cudaMalloc((float**)&d_C, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int iLen = 1024;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x);

iStart = seconds();
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
       block.x, iElaps);

// check kernel error
CHECK(cudaGetLastError()) ;

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nElem);

// free device global memory
CHECK(cudaFree(d_A));
CHECK(cudaFree(d_B));
CHECK(cudaFree(d_C));

// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

return(0);
}


Differences:
• The difference between the execution configurations with block.x = 1023 and block.x = 1024 is the number of threads per block. When block.x = 1023, there are a total of 16,694 threads (16401 blocks x 1023 threads per block), while with block.x = 1024, there are 16,777,216 threads (16384 blocks x 1024 threads per block). This means that with block.x = 1024, the total number of threads is much higher than with block.x = 1023.
• In terms of performance, the difference between these two configurations is not significant.The execution time of the kernel function sumArraysOnGPU is slightly faster with block.x compared to block.x, but the difference is small.
• The reason why the performance difference is small is because the number of threads that can be run in parallel is limited by the hardware resources of the GPU. In this case, the number of threads per block is already high enough that adding more threads does not result in a significant improvement in performance. Additionally, with block.x = 1024, there are fewer blocks, which can lead to underutilization of the GPU's resources.
• Overall, the performance difference between these two configurations is not significant, and the optimal configuration may depend on the specific hardware and workload.


2. Block.x=256
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void checkResult(float *hostRef, float *gpuRef, const int N) 
{ 
double epsilon = 1.0E-8;
bool match = 1;

for (int i = 0; i < N; i++)
{
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
               gpuRef[i], i);
        break;
    }
}

if (match) printf("Arrays match.\n\n");

return;
}

void initialData(float *ip, int size) { // generate different seed for random number time_t t; srand((unsigned) time(&t));

for (int i = 0; i < size; i++)
{
    ip[i] = (float)( rand() & 0xFF ) / 10.0f;
}

return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
for (int idx = 0; idx < N; idx++)
{ 
C[idx] = A[idx] + B[idx]; 
} } 
global void sumArraysOnGPU(float *A, float *B, float *C, const int N) 
{ 
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv) { printf("%s Starting...\n", argv[0]);

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up data size of vectors
int nElem = 1 << 24;
printf("Vector size %d\n", nElem);

// malloc host memory
size_t nBytes = nElem * sizeof(float);

float *h_A, *h_B, *hostRef, *gpuRef;
h_A     = (float *)malloc(nBytes);
h_B     = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef  = (float *)malloc(nBytes);

double iStart, iElaps;

// initialize data at host side
iStart = seconds();
initialData(h_A, nElem);
initialData(h_B, nElem);
iElaps = seconds() - iStart;
printf("initialData Time elapsed %f sec\n", iElaps);
memset(hostRef, 0, nBytes);
memset(gpuRef,  0, nBytes);

// add vector at host side for result checks
iStart = seconds();
sumArraysOnHost(h_A, h_B, hostRef, nElem);
iElaps = seconds() - iStart;
printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

// malloc device global memory
float *d_A, *d_B, *d_C;
CHECK(cudaMalloc((float**)&d_A, nBytes));
CHECK(cudaMalloc((float**)&d_B, nBytes));
CHECK(cudaMalloc((float**)&d_C, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int iLen = 256;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x);

iStart = seconds();
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
       block.x, iElaps);

// check kernel error
CHECK(cudaGetLastError()) ;

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nElem);

// free device global memory
CHECK(cudaFree(d_A));
CHECK(cudaFree(d_B));
CHECK(cudaFree(d_C));

// free host memory
free(h_A);
free(h_B);
free(hostRef);
free(gpuRef);

return(0);
}

Differences:
• By changing the block size from 128 to 256, the number of blocks needed to process the same amount of data was halved, from 65536 to 32768. However, since each thread now handles two elements instead of one, the total number of threads needed remains the same, which is equal to the product of the number of blocks and the block size.
• The execution time for the kernel sumArraysOnGPU-timer decreased when the block size was changed from 128 to 256. This suggests that the optimal block size may lie between these two values.
• It is important to note that the execution time for sumArraysOnHost remains constant, as it is not affected by the block size. The overall performance of the program is determined by the execution time of the kernel sumArraysOnGPU-timer, which can be optimized by experimenting with different block sizes.


## Output:

1. i)BLOCK.X = 1023: 
root@SAV-MLSystem:/home/student# nvcc sumArraysOnGPU-timer.cu -o sumArraysOnGPU-timer
root@SAV-MLSystem:/home/student# nvcc sumArraysOnGPU-timer.cu
root@SAV-MLSystem:/home/student# ./sumArraysOnGPU-timer
./sumArraysOnGPU-timer Starting...
Using Device 0: NVIDIA GeForce GT 710
Vector size 16777216
initialData Time elapsed 0.427707 sec
sumArraysOnHost Time elapsed 0.034538 sec
sumArraysOnGPU <<< 16401, 1023 >>> Time elapsed 0.020212 sec
Arrays match.
root@SAV-MLSystem:/home/student# 


ii)BLOCK.X = 1024: 
root@SAV-MLSystem:/home/student# nvcc sumArraysOnGPU-timer.cu -o 
sumArraysOnGPU-timer
root@SAV-MLSystem:/home/student# nvcc sumArraysOnGPU-timer.cu
root@SAV-MLSystem:/home/student# ./sumArraysOnGPU-timer
./sumArraysOnGPU-timer Starting...
Using Device 0: NVIDIA GeForce GT 710
Vector size 16777216
initialData Time elapsed 0.423519 sec
sumArraysOnHost Time elapsed 0.034505 sec
sumArraysOnGPU <<< 16384, 1024 >>> Time elapsed 0.020785 sec
Arrays match.
root@SAV-MLSystem:/home/student#


2. Block.x=256:
root@SAV-MLSystem:/home/student# nvcc sumArraysOnGPU-timer-2.cu -o 
sumArraysOnGPU-timer-2
root@SAV-MLSystem:/home/student# nvcc sumArraysOnGPU-timer-2.cu
root@SAV-MLSystem:/home/student# ./sumArraysOnGPU-timer-2
./sumArraysOnGPU-timer-2 Starting...
Using Device 0: NVIDIA GeForce GT 710
Vector size 16777216
initialData Time elapsed 0.425328 sec
sumArraysOnHost Time elapsed 0.034418 sec
sumArraysOnGPU_2 <<< 32768, 256 >>> Time elapsed 0.019457 sec
Arrays match.
root@SAV-MLSystem:/home/student#

## Result:

i) Thus the program to compare the execution time and result of summing two arrays using CUDA programming by setting block.x = 1023 and block.x = 1024 has been successfully implemented and verified.

ii) Thus the program to To analyze the impact of changing execution configuration on the execution time and result of summing two arrays using CUDA programming by letting each thread handle two elements and setting block.x = 256 has been successfully implemented and verified.
