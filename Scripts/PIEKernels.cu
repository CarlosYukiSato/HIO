
#include <pycuda-complex.hpp>
typedef pycuda::complex<float> complex;

#define NumModes X  // Replace with number of modes, otherwise this line will cause an error

__device__ inline float abs2(complex c)
{
	return c.real()*c.real() + c.imag()*c.imag();
}

__global__ void CopyFromROI(int size, complex* dest, complex* src, int ofy, int ofx, int roisizex, int objsizex)
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
		
		dest[index] = src[largeptr];
		index += numthreads;
	}
}

__global__ void UpdateProbeAndRspace(int size, complex* rspace, complex* exitwave, complex* buffer_exitwave, 
				complex* aperture, float betaObj, float Pmax2, float betaApert, float Omax2, float rpiealpha)
{
	int numthreads = blockDim.x*gridDim.x;
	int indexO = blockIdx.x*blockDim.x + threadIdx.x;

	while(indexO<size)
	{
		complex rsp = rspace[indexO];
		float APDEN = (1.0f-rpiealpha)*Omax2 + rpiealpha*abs2(rsp);
		complex invApertUpdate = betaApert*conj(rsp)/APDEN;
		
		for(int mode = 0; mode < NumModes; mode++)
		{			
			int indexP = indexO + mode*size;
			complex deltaPhi = exitwave[indexP] - buffer_exitwave[indexP];
		
			float ObjDenominator = (1.0f-rpiealpha)*Pmax2 + rpiealpha*abs2(aperture[indexP]);
			rsp += betaObj*deltaPhi*conj(aperture[indexP])/ObjDenominator;
			aperture[indexP] += invApertUpdate*deltaPhi;
		}
		rspace[indexO] = rsp;
		indexO += numthreads;
	}
	
}

__global__ void PhaseShift(int size, complex* kspace, float* kxx, float* kyy, float dx, float dy)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{
		float phase = kxx[index]*dx + kyy[index]*dy;
		kspace[index] *= complex(cos(phase),sin(phase));
		index += numthreads;
	}
}


__global__ void ReplaceInObject(int size, complex* finalObj,  complex* rspace, int ofy, int ofx, int roisizex, int objsizex)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{
		int j = int(index/roisizex);
		int i = index - j*roisizex;

		j += ofy;
		i += ofx;
		
		finalObj[j*objsizex + i] = rspace[index];
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
			float ksabs = 1E-20;

			for(int mode=0; mode < NumModes; mode++)
				ksabs += abs2(kspace[index + mode*size]);

			ksabs = sqrt(ksabs);
			errorsum += fabsf(ksabs-difpad[index]);
			ksabs = difpad[index]/ksabs;

			for(int mode=0; mode < NumModes; mode++)
				kspace[index + mode*size] *= ksabs;
		}
		index += numthreads;
	}
	error[gindex] = errorsum;
}

__global__ void ExitwaveAndBuffer(int size, complex* exitwave, complex* buffer_exitwave, complex* aperture, complex* rspace)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{
		complex rss = rspace[index];

		for(int mode=0; mode < NumModes; mode++)
		{
			int i = index + mode*size;
			pycuda::complex<float> res = rss*aperture[i];
			exitwave[i] = res;
			buffer_exitwave[i] = res;
		}
		index += numthreads;
	}
}


__global__ void CopyOffset(int size, complex* des, int ofd, complex*  src, int ofs)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{
		des[index+ofd] = src[index+ofs];
		index += numthreads;
	}
}


__global__ void ApertureAbs2(int size, complex* aperture, float* result)
{
	int numthreads = blockDim.x*gridDim.x;
	int gindex = blockIdx.x*blockDim.x + threadIdx.x;
	
	float regmax = 0;

	for(int index = gindex; index<size; index+=numthreads)
	{
		float res = 1E-10f;

		for(int mode=0; mode < NumModes; mode++)
			res += abs2(aperture[index + mode*size]);
			
		regmax = max(regmax,res);
	}
	result[gindex] = regmax;
}

__global__ void ObjectAbs2(int size, complex* object, float* result)
{
	int numthreads = blockDim.x*gridDim.x;
	int gindex = blockIdx.x*blockDim.x + threadIdx.x;
	
	float regmax = 0;
	
	for(int index = gindex; index<size; index+=numthreads)
		regmax = max(regmax,abs2(object[index]));
		
	result[gindex] = regmax;
}

__global__ void ScalarProd(int size, complex* aperture, complex* dot, int mode1, int mode2)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int dotindex = index;
	
	complex res = 0;
	complex* aperture1 = aperture + mode1*size;
	complex* aperture2 = aperture + mode2*size;
	
	for(; index<size; index+=numthreads)
		res += aperture1[index]*conj(aperture2[index]);
		
	dot[dotindex] = res;
}
__global__ void NormSquared(int size, complex* aperture, float* dot)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int dotindex = index;
	
	float res = 0;
	
	for(; index<NumModes*size; index+=numthreads)
		res += abs2(aperture[index]);
		
	dot[dotindex] = res;
}

__global__ void ModeSquared(int size, complex* aperture, float* dot, int mode)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int dotindex = index;
	
	float res = 0;
	complex* aperturemode = aperture + mode*size;
	
	for(; index<size; index+=numthreads)
		res += abs2(aperturemode[index]);
		
	dot[dotindex] = res;
}

__global__ void ModeSub21(int size, complex* aperture, int mode1, int mode2, complex alpha)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	complex* aperture1 = aperture + mode1*size;
	complex* aperture2 = aperture + mode2*size;
	
	for(; index<size; index+=numthreads)
		aperture2[index] -= alpha*aperture1[index];
}
__global__ void ModeMultiply(int size, complex* aperture, int mode, float alpha)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	complex* aperturemode = aperture + mode*size;
	
	for(; index<size; index+=numthreads)
		aperturemode[index] *= alpha;
}

__global__ void ProbeReduction(int size, complex* aperture, int side)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x + 2*side + 2;
	
	for(; index<size - 2*side - 2; index+=numthreads)
	{
		complex apindex = 0;
		
		for(int m=0; m<5; m++) 
			for(int n=0; n<5; n++)
				apindex += aperture[index + size*(5*m + n) + side*(m-2) + n - 2];
				
		apindex /= 25.0f;
				
		for(int m=0; m<5; m++)
			for(int n=0; n<5; n++)
				aperture[index + size*(5*m + n) + side*(m-2) + n - 2] = apindex;
	}
}


__global__ void ProbePartialReduction(int size, complex* aperture, int side)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x + 2*side + 2;
	
	for(; index<size - 2*side - 2; index+=numthreads)
	{
		complex apindex = 0;
		
		for(int m=0; m<5; m++) 
				apindex += aperture[index + size*(5+m) + side*(m-2)];
		for(int n=0; n<5; n++)
				apindex += aperture[index + size*n + n - 2];
				
		apindex /= 10.0f;
				
		for(int m=0; m<5; m++)
				aperture[index + size*(5+m) + side*(m-2)] = apindex;
		for(int n=0; n<5; n++)
				aperture[index + size*n + n - 2] = apindex;
	}
}

__global__ void CropObject(int size, complex* aperture, complex* object, int32_t* ROI, int numrois, 
				int objsizex, int roisizex, float probethresh, int objsize, float crop_factor)
{
	int numthreads = blockDim.x*gridDim.x;
	int objindex = blockIdx.x*blockDim.x + threadIdx.x;
	int roisizey = size/roisizex;
	
	for(; objindex < objsize; objindex += numthreads)
	{
		float sumprobes = 0;
		int oy = objindex/objsizex;
		int ox = objindex - oy*objsizex;
			
		for(int roi=0; roi<numrois; roi++)
		{
			int yi = oy - ROI[2*roi];
			int xi = ox - ROI[2*roi+1];
			
			if(yi >= 0 && yi < roisizey && xi >=0 && xi < roisizex)
				 for(int mode=0; mode<NumModes; mode++)
					sumprobes += abs2( aperture[yi*roisizex + xi + mode*size] );				
		}
			
		float Mask = sumprobes > probethresh ? 1.0f : crop_factor;
		object[objindex] *= Mask;
	}
}

__global__ void PhiDerivative(int size, complex* hx, complex* hy, complex* d0, complex* aperture, complex* rspace, int sizex)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i<size; i += numthreads)
	{
		if(i+sizex < size)
		{
			d0[i] = aperture[i]*rspace[i];
			hx[i] = aperture[i+1]*rspace[i];
			hy[i] = aperture[i+sizex]*rspace[i];
		}
		else
		{
			d0[i] = 0;
			hx[i] = 0;
			hy[i] = 0;
		}
	}
}

__global__ void PosReductions(int size, complex* HX, complex* HY, complex* d0, float* difpad, float* vReduction)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float X2 = 0, XY = 0, Y2 = 0, Xx = 0, Yy = 0;

	for(int i=index; i<size; i+=numthreads)
	{
		complex PhiConj = conj(d0[i]);

		float IX = 2.0f*(HX[i]*PhiConj).real();
		float IY = 2.0f*(HY[i]*PhiConj).real();
		
		float deltaI = difpad[i]*difpad[i] - abs2(PhiConj);
		
		X2 += IX*IX;
		Y2 += IY*IY;
		XY += IX*IY;	
		Xx += IX*deltaI;
		Yy += IY*deltaI;
	}
	
	vReduction[index + 0*numthreads] = X2;
	vReduction[index + 1*numthreads] = Y2;
	vReduction[index + 2*numthreads] = XY;
	vReduction[index + 3*numthreads] = Xx;
	vReduction[index + 4*numthreads] = Yy;
}

__global__ void PhaseGrad(int size, complex* ks, complex* kcopy, float* kmapx, float* kmapy, float dx, float dy)
{
	int numthreads = blockDim.x*gridDim.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(; i<size; i += numthreads)
	{
		float phase = kmapx[i]*dx + kmapy[i]*dy;
		kcopy[i] = complex(cos(phase),sin(phase)) * ks[i];
	}
}





