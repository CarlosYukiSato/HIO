
#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complex;

#define NumSlices X  // Replace with number of modes, otherwise this line will cause an error
#define ObjectSize 1024*1024

__device__ inline float abs2(complex c)
{
	return c.real()*c.real() + c.imag()*c.imag();
}

__global__ void UpdateProbeAndRspace(int size, complex* finalObj, complex* exitwave, complex* buffer_exitwave, 
				complex* aperture, int ofy, int ofx, int roisizex, int objsizex)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{
		int j = int(index/roisizex);
		int i = index - j*roisizex;
		
		j += ofy;
		i += ofx;

		int largeptr = j*objsizex + i;
				
		complex deltaPhi = exitwave[index] - buffer_exitwave[index];
		
		for(int sl = 0; sl < NumSlices; sl++)
			finalObj[largeptr+sl*ObjectSize] += 0.8f*deltaPhi*conj(aperture[index+sl*size]);
			
		index += numthreads;
	}
}


__global__ void ApplyDifPad(int size, complex* kspace, float* difpad, float* error)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	int gindex = index;
	float errorsum = 0;

	while(index<size)
	{
		if(difpad[index] >= 0)
		{
			float ksabs = 1E-20 + abs(kspace[index]);
			errorsum += fabsf(ksabs-difpad[index]);
			kspace[index] *= difpad[index]/ksabs;
		}
		index += numthreads;
	}
	error[gindex] = errorsum;
}

__global__ void ExitwaveAndBuffer(int size, complex* buffer_exitwave, complex* aperture, 
				complex* finalObj, int ofy, int ofx, int roisizex, int objsizex)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{		
		int j = int(index/roisizex);
		int i = index - j*roisizex;
		
		j += ofy;
		i += ofx;

		int largeptr = j*objsizex + i;
		complex BF = 0;
		
		for(int sl = 0; sl < NumSlices; sl++)
			BF += finalObj[largeptr+sl*ObjectSize]*aperture[index+sl*size];
		
		buffer_exitwave[index] = BF;
		index += numthreads;
	}
}




