#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complex;

#define NumModes X  // Replace with number of modes, otherwise this line will cause an error

__device__ inline float abs2(complex c)
{
	return c.real()*c.real() + c.imag()*c.imag();
}

__global__ void Error2Abs(int size, float* err, complex* rspace) 
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
		err[i] = abs2(rspace[i]);
}

__global__ void Error1DifCF(int size, complex* ktemp, float* difpad, float* kerror)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	float err = 0.0f;

	for(int i = index; i < size; i += numthreads)
	if(difpad[i] >= 0)
		err += fabsf( abs(ktemp[i]) - difpad[i] );
	
	kerror[index] = err;
}

__global__ void Error2DifCF(int size, complex* ktemp, float* difpad, float* kerror)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	
	if(difpad[i] >= 0)
	{
		float errabs = abs(ktemp[i]) - difpad[i];
		kerror[i] = errabs*errabs;
	}
}

__global__ void ZeroInMask(int size, complex* rspace, bool* Mask)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	
	if(Mask[i])
		rspace[i] = 0;
}

__global__ void ZeroOutMask(int size, complex* rspace, bool* Mask)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	
	if(!Mask[i])
		rspace[i] = 0;
}

__global__ void HIOStep(int size, complex* rspace, complex* kspace, complex* buffer, complex* sample, bool* Mask)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	{
		if(Mask[i])
			sample[i] = rspace[i];

		rspace[i] = buffer[i] - 0.9f*rspace[i];

		if(sample[i].real() < 0)
			sample[i] = rspace[i];

		if(Mask[i])
			rspace[i] = sample[i];

		buffer[i] = rspace[i];
	}
}

__global__ void Copy(int size, complex* x, complex* y)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
		x[i] = y[i]; // I dont trust python's assignment operator
}

__global__ void SGDStep(int size, complex* outsidefft, complex* kspace, float* prevPhasegrad, float* prevAmpgrad, bool* NegMask)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	{
		const float momentum = 0.9f;

		complex grad = 1E4f*kspace[i]*conj(outsidefft[i])/abs(kspace[i]);

		float gradimg = grad.imag();
		gradimg = min(gradimg,0.01f);
		gradimg = max(gradimg,-0.01f);

		float newgrad = prevPhasegrad[i]*momentum + gradimg;
		prevPhasegrad[i] = newgrad;

		kspace[i] *= complex(cos(newgrad),sin(newgrad));

		if(false) //if(NegMask[i])
		{
			newgrad = 1E3f*grad.real()/float(abs(kspace[i])+1E-10) + prevAmpgrad[i]*0.95f;
	
			kspace[i] -= kspace[i]*newgrad;
			prevAmpgrad[i] = newgrad;
		}
	}
}

__global__ void FastSGDStep(int size, complex* outsidefft, complex* kspace, float* prevPhasegrad, float* prevAmpgrad, bool* NegMask)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	{
		complex grad = 2E-4f*kspace[i]*conj(outsidefft[i])/abs(kspace[i]);
		float newgrad = grad.imag();
	
		if(prevPhasegrad[i]*newgrad>0.0f)
		{
			if(fabsf(prevPhasegrad[i]) < 1E4f)
				prevPhasegrad[i] *= 3.0f;
		}
		else
		{
			prevPhasegrad[i] *= -0.2f;
			if(fabsf(prevPhasegrad[i]) < 1.0f)
				prevPhasegrad[i] /= fabsf(prevPhasegrad[i]);
		}

		newgrad *= fabsf(prevPhasegrad[i]);

		newgrad = min(newgrad,0.25f);
		newgrad = max(newgrad,-0.25f);

		prevAmpgrad[i] = newgrad;

		kspace[i] *= complex(cos(newgrad),sin(newgrad));

		if(NegMask[i])
		{
			newgrad = grad.real() + 0.8f*prevAmpgrad[i];
			kspace[i] -= 0.5f*kspace[i]*newgrad;

			prevAmpgrad[i] = newgrad;
		}
	}
}

__global__ void SGDParabolic(int size, complex* outsidefft, complex* kspace, float* prevPhasegrad, float* prevAmpgrad, bool* NegMask)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	{
		complex complexgrad = kspace[i]*conj(outsidefft[i])*1E-7f;
		float grad = complexgrad.imag();
		float secgrad = complexgrad.real()*0.5f;

		float secdev = grad - prevPhasegrad[i];
		secdev /= prevAmpgrad[i];

		float xmin = -grad/secdev;
		if(fabsf(xmin) > 0.1f)
			xmin /= fabsf(xmin);

		xmin *= 0.25f;

		prevPhasegrad[i] = grad;
		prevAmpgrad[i] = xmin;

		kspace[i] *= complex(cos(xmin),sin(xmin));
	}
}

__global__ void ApplyDifPad(int size, complex* kspace, float* difpad)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	
	if(difpad[i] >= 0)
	{
		float ksabs = abs(kspace[i]) + 1E-10;
		kspace[i] *= difpad[i]/ksabs;
	}
}

__global__ void PhaseShift(int size, complex* image, float* kx, float* ky, float dx, float dy)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	{
		float exponent = kx[i]*dx + ky[i]*dy;
		complex rot = complex(cos(exponent),sin(exponent));
		image[i] *= rot;
	}
}

__global__ void PhiDerivative(int size, complex* hx, complex* hy, complex* aperture, complex* rspace, complex* d0, int sizex)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	{
		d0[i] = aperture[i]*rspace[i];

	
		if(i+1 < size)
			hx[i] = aperture[i+1]*rspace[i];
		else
			hx[i] = 0.0f;

		if(i+sizex<size)
			hy[i] = aperture[i+sizex]*rspace[i];
		else
			hy[i] = 0.0f;
	}
}
		
__global__ void IntensityDerivative(int size, complex* HX, complex* HY, complex* d0, float* IX, float* IY)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i < size; i += numthreads)
	{
		complex Phi = d0[i];
		complex PhiConj = conj(Phi);
		float PhiSq = (Phi*PhiConj).real();

		IX[i] = 2.0f*( PhiSq - (HX[i]*PhiConj).real() );
		IY[i] = 2.0f*( PhiSq - (HY[i]*PhiConj).real() );
	}
}



