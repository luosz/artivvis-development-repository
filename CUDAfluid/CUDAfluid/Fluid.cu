#include "Fluid.h"

void Fluid::Init()
{
	simData.Init();
	tempData.Init();
	AMatrix.Init();

	hostDensities.resize(numGridCells);
	hostTemperatures.resize(numGridCells);

	HANDLE_ERROR( cudaMalloc((void**)&vortX, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&vortY, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&vortZ, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&vortMag, numGridCells * sizeof(float)) );
}


__device__ float GetDataIJK(int i, int j, int k, int *res, float *data, int axis)
{
	if (axis == 0)
	{
		if (i < 0 || i >= res[0])
			return 0.0f;

		j = glm::max(j, 0);
		j = glm::min(j, res[1] - 1);
		
		k = glm::max(k, 0);
		k = glm::min(k, res[2] - 1);
	}
	else if (axis == 1)
	{
		if (j < 0 || j >= res[1])
			return 0.0f;

		i = glm::max(i, 0);
		i = glm::min(i, res[0] - 1);
		
		k = glm::max(k, 0);
		k = glm::min(k, res[2] - 1);
	}
	else if (axis == 2)
	{
		if (k < 0 || k >= res[2])
			return 0.0f;

		i = glm::max(i, 0);
		i = glm::min(i, res[0] - 1);

		j = glm::max(j, 0);
		j = glm::min(j, res[1] - 1);
	}
	else if (axis == 3)
	{
		if (i < 0 || j < 0 || k < 0 || i >= res[0] || j >= res[1] || k >= res[2])
			return 0.0f;
	}
	
	return data[i + (j * res[0]) + (k * res[0] * res[1])];
}

__device__ void SetDataIJK(int i, int j, int k, int *res, float *data, float value, int axis)
{	
	if (axis == 0)
	{
		if (i < 0 || i >= res[0])
			return;

		j = glm::max(j, 0);
		j = glm::min(j, res[1] - 1);
		
		k = glm::max(k, 0);
		k = glm::min(k, res[2] - 1);
	}
	else if (axis == 1)
	{
		if (j < 0 || j >= res[1])
			return;

		i = glm::max(i, 0);
		i = glm::min(i, res[0] - 1);
		
		k = glm::max(k, 0);
		k = glm::min(k, res[2] - 1);
	}
	else if (axis == 2)
	{
		if (k < 0 || k >= res[2])
			return;

		i = glm::max(i, 0);
		i = glm::min(i, res[0] - 1);

		j = glm::max(j, 0);
		j = glm::min(j, res[1] - 1);
	}
	else if (axis == 3)
	{
		if (i < 0 || j < 0 || k < 0 || i >= res[0] || j >= res[1] || k >= res[2])
			return;
	}

	data[i + (j * res[0]) + (k * res[0] * res[1])] = value;
}

__device__ glm::vec3 GetCentre(int i, int j, int k, float dx)
{
	return glm::vec3((i+0.5f) * dx, (j+0.5f) * dx, (k+0.5f) * dx);
}


__global__ void CudaAddArtificialForces(SimData data)
{
	int span = 15;
	int middleX = data.gridRes[0] / 2;
	int middleY = data.gridRes[1] / 2;
	int middleZ = data.gridRes[2] / 2;

	for (int i=-span; i<=span; i++)
	{
		for (int k=-span; k<=span; k++)
		{
			if (glm::distance(glm::vec2(i, k), glm::vec2(0, 0)) > span)
				continue;

			SetDataIJK(middleX + i,	0,	middleZ + k, data.gridRes, data.densities, 1.5f, 3);
			SetDataIJK(middleX + i,	0,	middleZ + k, data.gridRes, data.temperatures, 600.0f, 3);
		}
	}

//	for (int j=-span; j<=span; j++)
//	{
//		for (int k=-span; k<=span; k++)
//		{
//			if (glm::distance(glm::vec2(j, k), glm::vec2(0, 0)) > span)
//				continue;
//
//			SetDataIJK(0, middleY + j,	middleZ + k, data.gridRes, data.densities, 1.0f, 3);
//			SetDataIJK(0, middleY + j,	middleZ + k, data.gridRes, data.temperatures, 0.0f, 3);
//		}
//	}
//
//	for (int i=0; i<data.gridRes[0] / 4; i++)
//	{
//		int coneCoefficient = ((float)i / (float)data.gridRes[0]) * span;
//		coneCoefficient = 0;
//
//		for (int j=-span+coneCoefficient; j<=span-coneCoefficient; j++)
//		{
//			for (int k=-span+coneCoefficient; k<=span-coneCoefficient; k++)
//			{
//				if (glm::distance(glm::vec2(j, k), glm::vec2(0, 0)) > span)
//					continue;
//
//				SetDataIJK(i, middleY + j,	middleZ + k, data.xFaceRes, data.xVelocities, 10.0f, 0);
//			}
//		}
//	}
}

void Fluid::Update()
{
	CudaAddArtificialForces<<<1, 1>>>(simData);

	AdvectVelocity();
	AddExternalForces();

	Project();

	AdvectTemperature();
	AdvectDensity();

	HANDLE_ERROR( cudaMemcpy(&hostDensities[0], simData.densities, numGridCells * sizeof(float), cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&hostTemperatures[0], simData.temperatures, numGridCells * sizeof(float), cudaMemcpyDeviceToHost) );
	cudaDeviceSynchronize();
}


#pragma region CubicInterp

__device__ float CubicInterpolate(float fkm1, float fk, float fkp1, float fkp2, float fract)
{
	if(fract == 0)
		return fk;

	float dk = 0.5f * (fkp1 - fkm1);
	float dkp1 = 0.5f  *(fkp2 - fk);
	float deltak = fkp1 - fk;

	if(deltak != 0.0f)
	{
		if(deltak < 0.0f)
		{
			if(dk > 0.0f) 
				dk = 0.0f;
			if(dkp1 > 0.0f) 
				dkp1 = 0.0f;
		}
		else
		{
			if(dk < 0.0f) 
				dk = 0.0f;
			if(dkp1 < 0.0f) 
				dkp1 = 0.0f;
		}
	}
	else
	{
		dk = 0.0f;
		dkp1 = 0.0f;
	}

	return (dk + dkp1 - 2.0f * deltak) * fract * fract * fract + (3.0f * deltak - 2.0f * dk - dkp1) * fract * fract + dk * fract + fk;
}

__device__ float CubicInterpolateY(int i, int j, int k, float *vector, int *res, int axis, float yFraction)
{
	float tmp1 = GetDataIJK(i, j-1, k, res, vector, axis);
	float tmp2 = GetDataIJK(i, j, k, res, vector, axis);
	float tmp3 = GetDataIJK(i, j+1, k, res, vector, axis);
	float tmp4 = GetDataIJK(i, j+2, k, res, vector, axis);;

	return CubicInterpolate(tmp1, tmp2, tmp3, tmp4, yFraction);
}

__device__ float Interpolate2(glm::vec3 pos, float *vector, int *res, float dx, int axis)
{
	if (axis == 0)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y - dx*0.5f), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z - dx*0.5f), (float)res[2]);
	}
	else if (axis == 1)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x - dx*0.5f), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z - dx*0.5f), (float)res[2]);
	}
	else if (axis == 2)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x - dx*0.5f), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y - dx*0.5f), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z), (float)res[2]);
	}
	if (axis == 3)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x - dx*0.5f), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y - dx*0.5f), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z - dx*0.5f), (float)res[2]);
	}

	int i = (int) pos.x / dx;
	int j = (int) pos.y / dx;
	int k = (int) pos.z / dx;

	float xFraction = (pos.x - (i*dx)) / dx;
	float yFraction = (pos.y - (j*dx)) / dx;
	float zFraction = (pos.z - (k*dx)) / dx;

	float tmp1 = CubicInterpolateY(i-1, j, k-1, vector, res, axis, yFraction);
	float tmp2 = CubicInterpolateY(i, j, k-1, vector, res, axis, yFraction);
	float tmp3 = CubicInterpolateY(i+1, j, k-1, vector, res, axis, yFraction);
	float tmp4 = CubicInterpolateY(i+2, j, k-1, vector, res, axis, yFraction);
	float tmp5 = CubicInterpolate(tmp1,tmp2,tmp3,tmp4,xFraction);

	tmp1 = CubicInterpolateY(i-1, j, k, vector, res, axis, yFraction);
	tmp2 = CubicInterpolateY(i, j, k, vector, res, axis, yFraction);
	tmp3 = CubicInterpolateY(i+1, j, k, vector, res, axis, yFraction);
	tmp4 = CubicInterpolateY(i+2, j, k, vector, res, axis, yFraction);
	float tmp6 = CubicInterpolate(tmp1,tmp2,tmp3,tmp4,xFraction);

	tmp1 = CubicInterpolateY(i-1, j, k+1, vector, res, axis, yFraction);
	tmp2 = CubicInterpolateY(i, j, k+1, vector, res, axis, yFraction);
	tmp3 = CubicInterpolateY(i+1, j, k+1, vector, res, axis, yFraction);
	tmp4 = CubicInterpolateY(i+2, j, k+1, vector, res, axis, yFraction);
	float tmp7 = CubicInterpolate(tmp1,tmp2,tmp3,tmp4,xFraction);

	tmp1 = CubicInterpolateY(i-1, j, k+2, vector, res, axis, yFraction);
	tmp2 = CubicInterpolateY(i, j, k+2, vector, res, axis, yFraction);
	tmp3 = CubicInterpolateY(i+1, j, k+2, vector, res, axis, yFraction);
	tmp4 = CubicInterpolateY(i+2, j, k+2, vector, res, axis, yFraction);
	float tmp8 = CubicInterpolate(tmp1,tmp2,tmp3,tmp4,xFraction);

	double tmp9 = CubicInterpolate(tmp5,tmp6,tmp7,tmp8,zFraction);

	return tmp9;
}

#pragma endregion CubicInterp


#pragma region VelocityStuff

__device__ float LERP(float val1, float val2, float fraction)
{
	return ((1.0f - fraction) * val1) + (fraction * val2);
}

__device__ float Interpolate(glm::vec3 pos, float *vector, int *res, float dx, int axis)
{
	if (axis == 0)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y - dx*0.5f), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z - dx*0.5f), (float)res[2]);
	}
	else if (axis == 1)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x - dx*0.5f), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z - dx*0.5f), (float)res[2]);
	}
	else if (axis == 2)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x - dx*0.5f), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y - dx*0.5f), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z), (float)res[2]);
	}
	if (axis == 3)
	{
		pos.x = glm::min(glm::max(0.0f, pos.x - dx*0.5f), (float)res[0]);
		pos.y = glm::min(glm::max(0.0f, pos.y - dx*0.5f), (float)res[1]);
		pos.z = glm::min(glm::max(0.0f, pos.z - dx*0.5f), (float)res[2]);
	}

	int i = (int) pos.x / dx;
	int j = (int) pos.y / dx;
	int k = (int) pos.z / dx;

	float xFraction = (pos.x - (i*dx)) / dx;
	float yFraction = (pos.y - (j*dx)) / dx;
	float zFraction = (pos.z - (k*dx)) / dx;

	float v1 = GetDataIJK(i, j, k, res, vector, axis);
	float v2 = GetDataIJK(i, j+1, k, res, vector, axis);
	
	float v3 = GetDataIJK(i+1, j, k, res, vector, axis);
	float v4 = GetDataIJK(i+1, j+1, k, res, vector, axis);

	float v5 = GetDataIJK(i, j, k+1, res, vector, axis);
	float v6 = GetDataIJK(i, j+1, k+1, res, vector, axis);

	float v7 = GetDataIJK(i+1, j, k+1, res, vector, axis);
	float v8 = GetDataIJK(i+1, j+1, k+1, res, vector, axis);

	float tmp12 = LERP(v1, v2, yFraction);
	float tmp34 = LERP(v3, v4, yFraction);

	float tmp56 = LERP(v5, v6, yFraction);
	float tmp78 = LERP(v7, v8, yFraction);

	float tmp1234 = LERP(tmp12, tmp34, xFraction);
	float tmp5678 = LERP(tmp56, tmp78, xFraction);

	float tmp = LERP(tmp1234, tmp5678, zFraction);

	return tmp;
}


__device__ float GetXVelocity(glm::vec3 pos, SimData data)
{
	return Interpolate(pos, data.xVelocities, data.xFaceRes, data.dx[0], 0);
}

__device__ float GetYVelocity(glm::vec3 pos, SimData data)
{
	return Interpolate(pos, data.yVelocities, data.yFaceRes, data.dx[0], 1);
}

__device__ float GetZVelocity(glm::vec3 pos, SimData data)
{
	return Interpolate(pos, data.zVelocities, data.zFaceRes, data.dx[0], 2);
}

__device__ glm::vec3 GetVelocity(glm::vec3 pos, SimData data)
{
	glm::vec3 velocity(0.0f);

	velocity.x = GetXVelocity(pos, data);
	velocity.y = GetYVelocity(pos, data);
	velocity.z = GetZVelocity(pos, data);

	return velocity;
}


__global__ void CudaAdvectXVel(SimData tempData, SimData simData)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < simData.numFaces[0])
	{
		int remainder;
		int k = (int)(tid / (simData.xFaceRes[0] * simData.xFaceRes[1]));
		remainder = tid % (simData.xFaceRes[0] * simData.xFaceRes[1]);

		int j = (int)(remainder / simData.xFaceRes[0]);
		remainder = remainder % simData.xFaceRes[0];

		int i = remainder;

		if(i > 0 && i < (simData.xFaceRes[0] - 1))
		{
			glm::vec3 posVx = glm::vec3(i * simData.dx[0], (j+0.5f) * simData.dx[0], (k+0.5f) * simData.dx[0]);
			glm::vec3 u = GetVelocity(posVx, simData);
			glm::vec3 prevPosx = posVx - (0.5f * simData.dt[0] * u);

			u = GetVelocity(prevPosx, simData);
			prevPosx = posVx - (simData.dt[0] * u);

			tempData.xVelocities[tid] = GetXVelocity(prevPosx, simData);
		}
		else
			tempData.xVelocities[tid] = 0.0f;
	}
}

__global__ void CudaAdvectYVel(SimData tempData, SimData simData)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < simData.numFaces[1])
	{
		int remainder;
		int k = (int)(tid / (simData.yFaceRes[0] * simData.yFaceRes[1]));
		remainder = tid % (simData.yFaceRes[0] * simData.yFaceRes[1]);

		int j = (int)(remainder / simData.yFaceRes[0]);
		remainder = remainder % simData.yFaceRes[0];

		int i = remainder;

		if(j > 0 && j < (simData.yFaceRes[1] - 1))
		{

			glm::vec3 posVy = glm::vec3((i+0.5f) * simData.dx[0], j * simData.dx[0], (k+0.5f) * simData.dx[0]);
			glm::vec3 u = GetVelocity(posVy, simData);
			glm::vec3 prevPosy = posVy - (0.5f * simData.dt[0] * u);

			u = GetVelocity(prevPosy, simData);
			prevPosy = posVy - (simData.dt[0] * u);

			tempData.yVelocities[tid] = GetYVelocity(prevPosy, simData);

		}
		else
			tempData.yVelocities[tid] = 0.0f;
	}
}

__global__ void CudaAdvectZVel(SimData tempData, SimData simData)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < simData.numFaces[2])
	{
		int remainder;
		int k = (int)(tid / (simData.zFaceRes[0] * simData.zFaceRes[1]));
		remainder = tid % (simData.zFaceRes[0] * simData.zFaceRes[1]);

		int j = (int)(remainder / simData.zFaceRes[0]);
		remainder = remainder % simData.zFaceRes[0];

		int i = remainder;

		if(k > 0 && k < (simData.zFaceRes[2] - 1))
		{

			glm::vec3 posVz = glm::vec3((i+0.5f) * simData.dx[0], (j+0.5f) * simData.dx[0], k * simData.dx[0]);
			glm::vec3 u = GetVelocity(posVz, simData);
			glm::vec3 prevPosz = posVz - (0.5f * simData.dt[0] * u);

			u = GetVelocity(prevPosz, simData);
			prevPosz = posVz - (simData.dt[0] * u);

			tempData.zVelocities[tid] = GetZVelocity(prevPosz, simData);
		}
		else
			tempData.zVelocities[tid] = 0.0f;

	}
}


void Fluid::AdvectVelocity()
{
	HANDLE_ERROR( cudaMemcpy(tempData.xVelocities, simData.xVelocities, numXFaces * sizeof(float), cudaMemcpyDeviceToDevice) );
	HANDLE_ERROR( cudaMemcpy(tempData.yVelocities, simData.yVelocities, numYFaces * sizeof(float), cudaMemcpyDeviceToDevice) );
	HANDLE_ERROR( cudaMemcpy(tempData.zVelocities, simData.zVelocities, numZFaces * sizeof(float), cudaMemcpyDeviceToDevice) ); 

	CudaAdvectXVel<<<(numXFaces + 255) / 256, 256>>>(tempData, simData);
	CudaAdvectYVel<<<(numYFaces + 255) / 256, 256>>>(tempData, simData);
	CudaAdvectZVel<<<(numZFaces + 255) / 256, 256>>>(tempData, simData);

	HANDLE_ERROR( cudaMemcpy(simData.xVelocities, tempData.xVelocities, numXFaces * sizeof(float), cudaMemcpyDeviceToDevice) );
	HANDLE_ERROR( cudaMemcpy(simData.yVelocities, tempData.yVelocities, numYFaces * sizeof(float), cudaMemcpyDeviceToDevice) );
	HANDLE_ERROR( cudaMemcpy(simData.zVelocities, tempData.zVelocities, numZFaces * sizeof(float), cudaMemcpyDeviceToDevice) );
}

#pragma endregion VelocityStuff


#pragma region ExternalForces

void Fluid::AddExternalForces()
{
	CalculateBuoyancy();
	CalculateVorticity();
}


__global__ void CudaCalculateBuoyancy(SimData data)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < data.numFaces[1])
	{
		int remainder;
		int k = (int)(tid / (data.yFaceRes[0] * data.yFaceRes[1]));
		remainder = tid % (data.yFaceRes[0] * data.yFaceRes[1]);

		int j = (int)(remainder / data.yFaceRes[0]);
		remainder = remainder % data.yFaceRes[0];

		int i = remainder;

		if (j < 1 || j >= data.yFaceRes[1]-1)
			return;

		float ambientT = 300;
		float alpha = 0.1f;
		float beta = 0.4f;

		float currT, nextT, currS, nextS;

		currT = GetDataIJK(i, j, k, data.gridRes, data.temperatures, 3);
		nextT = GetDataIJK(i, j-1, k, data.gridRes, data.temperatures, 3);

		currS = GetDataIJK(i, j, k, data.gridRes, data.densities, 3);
		nextS = GetDataIJK(i, j-1, k, data.gridRes, data.densities, 3);

		float avgT = (currT + nextT) / 2.0f;
		float avgS = (currS + nextS) / 2.0f;

		float FbY = (-alpha * avgS) + (beta * (avgT - ambientT));

		data.yVelocities[tid] += data.dt[0] * FbY;
	}
}

void Fluid::CalculateBuoyancy()
{
	CudaCalculateBuoyancy<<<(numYFaces + 255) / 256, 256>>>(simData);
}


__global__ void CudaCalculateGradient(SimData data, float *vortX, float *vortY, float *vortZ, float *vortMag)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < data.numCells[0])
	{
		float dblDx = data.dx[0] * 2.0f;

		int remainder;
		int k = (int)(tid / (data.gridRes[0] * data.gridRes[1]));
		remainder = tid % (data.gridRes[0] * data.gridRes[1]);

		int j = (int)(remainder / data.gridRes[0]);
		remainder = remainder % data.gridRes[0];

		int i = remainder;


		//Accessors for adjacent cubes
		glm::vec3 iPrev = GetCentre(i-1, j, k, data.dx[0]);
		glm::vec3 iNext = GetCentre(i+1, j, k, data.dx[0]);
						  		 	
		glm::vec3 jPrev = GetCentre(i, j-1, k, data.dx[0]);
		glm::vec3 jNext = GetCentre(i, j+1, k, data.dx[0]);
						  		 	
		glm::vec3 kPrev = GetCentre(i, j, k-1, data.dx[0]);
		glm::vec3 kNext = GetCentre(i, j, k+1, data.dx[0]);

		//X-gradient component
		float x1 = GetZVelocity(jNext, data) - GetZVelocity(jPrev, data);
		float x2 = GetYVelocity(kNext, data) - GetYVelocity(kPrev, data);
		float xVort = (x1 - x2) / dblDx;

		//Y-gradient component
		float y1 = GetXVelocity(kNext, data) - GetXVelocity(kPrev, data);
		float y2 = GetZVelocity(iNext, data) - GetZVelocity(iPrev, data);
		float yVort = (y1 - y2) / dblDx;

		//Z-gradient component
		float z1 = GetYVelocity(iNext, data) - GetYVelocity(iPrev, data);
		float z2 = GetXVelocity(jNext, data) - GetXVelocity(jPrev, data);
		float zVort = (z1 - z2) / dblDx;

		//Vorticity Vector
		glm::vec3 vortOmega = glm::vec3(xVort, yVort, zVort);

		//Vorticity Magnitude
		float mag = glm::length(vortOmega);

		SetDataIJK(i, j, k, data.gridRes, vortX, xVort, 3);
		SetDataIJK(i, j, k, data.gridRes, vortY, yVort, 3);
		SetDataIJK(i, j, k, data.gridRes, vortZ, zVort, 3);
		SetDataIJK(i, j, k, data.gridRes, vortMag, mag, 3);
	}
}

__global__ void CudaCalculateVorticity(SimData data, float *vortX, float *vortY, float *vortZ, float *vortMag)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < data.numCells[0])
	{
		int remainder;
		int k = (int)(tid / (data.gridRes[0] * data.gridRes[1]));

		if (k < 1 || k >= (data.gridRes[2]-1))
			return;

		remainder = tid % (data.gridRes[0] * data.gridRes[1]);
		int j = (int)(remainder / data.gridRes[0]);

		if (j < 1 || j >= (data.gridRes[1]-1))
			return;

		remainder = remainder % data.gridRes[0];
		int i = remainder;

		if (i < 1 || i >= (data.gridRes[0]-1))
			return;


		float ep = 3.0f;	
		float dblDx = data.dx[0] * 2.0f;

		float gX = (GetDataIJK(i+1, j, k, data.gridRes, vortMag, 3) - GetDataIJK(i-1, j, k, data.gridRes, vortMag, 3)) / (dblDx);
		float gY = (GetDataIJK(i, j+1, k, data.gridRes, vortMag, 3) - GetDataIJK(i, j-1, k, data.gridRes, vortMag, 3)) / (dblDx);
		float gZ = (GetDataIJK(i, j, k+1, data.gridRes, vortMag, 3) - GetDataIJK(i, j, k-1, data.gridRes, vortMag, 3)) / (dblDx);

		//Gradient vector
		glm::vec3 grad = glm::vec3(gX, gY, gZ);

		//Normal (i,j,k)
		glm::vec3 N = grad / (glm::length(grad) + 10.0e-20f);

		//Omega
		glm::vec3 omega = glm::vec3(vortX[tid], vortY[tid], vortZ[tid]);

		//Force
		glm::vec3 Fconf = ep * data.dx[0] * glm::cross(N, omega);

		//Update Velocities with the vorticity confinement force
		float value = GetDataIJK(i, j, k, data.xFaceRes, data.xVelocities, 0) + data.dt[0] * Fconf.x * 0.5f;
		SetDataIJK(i, j, k, data.xFaceRes, data.xVelocities, value, 0);

		value = GetDataIJK(i+1, j, k, data.xFaceRes, data.xVelocities, 0) + data.dt[0] * Fconf.x * 0.5f;
		SetDataIJK(i+1, j, k, data.xFaceRes, data.xVelocities, value, 0);

		value = GetDataIJK(i, j, k, data.yFaceRes, data.yVelocities, 1) + data.dt[0] * Fconf.y * 0.5f;
		SetDataIJK(i, j, k, data.yFaceRes, data.yVelocities, value, 1);												// This needs to be fixed

		value = GetDataIJK(i, j+1, k, data.yFaceRes, data.yVelocities, 1) + data.dt[0] * Fconf.y * 0.5f;
		SetDataIJK(i, j+1, k, data.yFaceRes, data.yVelocities, value, 1);

		value = GetDataIJK(i, j, k, data.zFaceRes, data.zVelocities, 2) + data.dt[0] * Fconf.z * 0.5f;
		SetDataIJK(i, j, k, data.zFaceRes, data.zVelocities, value, 2);	

		value = GetDataIJK(i, j, k+1, data.zFaceRes, data.zVelocities, 2) + data.dt[0] * Fconf.z * 0.5f;
		SetDataIJK(i, j, k+1, data.zFaceRes, data.zVelocities, value, 2);	
	}
}


void Fluid::CalculateVorticity()
{
	CudaCalculateGradient<<<(numGridCells + 255) / 256, 256>>>(simData, vortX, vortY, vortZ, vortMag);

	CudaCalculateVorticity<<<(numGridCells + 255) / 256, 256>>>(simData, vortX, vortY, vortZ, vortMag);
}

#pragma endregion ExternalForces


#pragma region Project

__global__ void CudaCalculateXVelChange(SimData data)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < data.numFaces[0])
	{
		float rho = 1.0f;

		int remainder;
		int k = (int)(tid / (data.xFaceRes[0] * data.xFaceRes[1]));
		remainder = tid % (data.xFaceRes[0] * data.xFaceRes[1]);

		int j = (int)(remainder / data.xFaceRes[0]);
		remainder = remainder % data.xFaceRes[0];

		int i = remainder;

		if(i > 0 && i < (data.xFaceRes[0] - 1))
		{
			float currP = GetDataIJK(i, j, k, data.gridRes, data.pressures, 3);
			float prevP = GetDataIJK(i-1, j, k, data.gridRes, data.pressures, 3);
			
			float dPx = (data.dt[0] * (currP - prevP)) / (data.dx[0] * rho);
			data.xVelocities[tid] -= dPx;
		}
		else
		{
			data.xVelocities[tid] = 0.0f;
		}
	}
}

__global__ void CudaCalculateYVelChange(SimData data)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < data.numFaces[1])
	{
		float rho = 1.0f;

		int remainder;
		int k = (int)(tid / (data.yFaceRes[0] * data.yFaceRes[1]));
		remainder = tid % (data.yFaceRes[0] * data.yFaceRes[1]);

		int j = (int)(remainder / data.yFaceRes[0]);
		remainder = remainder % data.yFaceRes[0];

		int i = remainder;

		if(j > 0 && j < (data.yFaceRes[1] - 1))
		{
			float currP = GetDataIJK(i, j, k, data.gridRes, data.pressures, 3);
			float prevP = GetDataIJK(i, j-1, k, data.gridRes, data.pressures, 3);
			
			float dPy = (data.dt[0] * (currP - prevP)) / (data.dx[0] * rho);
			data.yVelocities[tid] -= dPy;
		}
		else
		{
			data.yVelocities[tid] = 0.0f;
		}
	}
}

__global__ void CudaCalculateZVelChange(SimData data)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < data.numFaces[2])
	{
		float rho = 1.0f;

		int remainder;
		int k = (int)(tid / (data.zFaceRes[0] * data.zFaceRes[1]));
		remainder = tid % (data.zFaceRes[0] * data.zFaceRes[1]);

		int j = (int)(remainder / data.zFaceRes[0]);
		remainder = remainder % data.zFaceRes[0];

		int i = remainder;

		if(k > 0 && k < (data.zFaceRes[2] - 1))
		{
			float currP = GetDataIJK(i, j, k, data.gridRes, data.pressures, 3);
			float prevP = GetDataIJK(i, j, k-1, data.gridRes, data.pressures, 3);
			
			float dPz = (data.dt[0] * (currP - prevP)) / (data.dx[0] * rho);
			data.zVelocities[tid] -= dPz;
		}
		else
		{
			data.zVelocities[tid] = 0.0f;
		}
	}
}

__global__ void CudaCalculateDiv(SimData data, thrust::device_ptr<float> divergence)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < data.numCells[0])
	{
		int remainder;
		int k = (int)(tid / (data.gridRes[0] * data.gridRes[1]));
		remainder = tid % (data.gridRes[0] * data.gridRes[1]);

		int j = (int)(remainder / data.gridRes[0]);
		remainder = remainder % data.gridRes[0];

		int i = remainder;

		float rho = 1.0f;
		float multiplier = -(((data.dx[0] * data.dx[0]) * rho) / data.dt[0]);
		float vBoundary = 0.0f;
		float divU, divV, divW;

		if (i == (data.gridRes[0] - 1)) 
		{
			divU = (vBoundary - GetDataIJK(i, j, k, data.xFaceRes, data.xVelocities, 0)) / data.dx[0]; 
		} 
		else if (i == 0) 
		{
			divU = (GetDataIJK(i+1, j, k, data.xFaceRes, data.xVelocities, 0) - vBoundary) / data.dx[0];
		} 
		else 
		{
			divU = (GetDataIJK(i+1, j, k, data.xFaceRes, data.xVelocities, 0) - GetDataIJK(i, j, k, data.xFaceRes, data.xVelocities, 0)) / data.dx[0];
		}



		if (j == (data.gridRes[1] - 1)) 
		{
			divV =  (vBoundary - GetDataIJK(i, j, k, data.yFaceRes, data.yVelocities, 1)) / data.dx[0];
		} 
		else if (j == 0 ) 
		{
			divV = (GetDataIJK(i, j+1, k, data.yFaceRes, data.yVelocities, 1) - vBoundary) / data.dx[0];
		} 
		else 
		{
			divV = (GetDataIJK(i, j+1, k, data.yFaceRes, data.yVelocities, 1) - GetDataIJK(i, j, k, data.yFaceRes, data.yVelocities, 1)) / data.dx[0]; 
		}

		if (k == (data.gridRes[2] - 1)) 
		{
			divW =  (vBoundary - GetDataIJK(i, j, k, data.zFaceRes, data.zVelocities, 2)) / data.dx[0];
		} 
		else if (k == 0 ) 
		{
			divW = (GetDataIJK(i, j, k+1, data.zFaceRes, data.zVelocities, 2) - vBoundary) / data.dx[0];
		} 
		else 
		{
			divW = (GetDataIJK(i, j, k+1, data.zFaceRes, data.zVelocities, 2) - GetDataIJK(i, j, k, data.zFaceRes, data.zVelocities, 2)) / data.dx[0]; 
		}


		float div = multiplier*(divU + divV + divW);
		divergence[tid] = div;
	}
}

void Fluid::Project()
{
	thrust::device_vector<float> divergence(numGridCells);

	CudaCalculateDiv<<<(numGridCells + 255) / 256, 256>>>(simData, &divergence[0]);

	CellDataMatrix A = this->AMatrix;

	int maxIterations = 100;
	float tolerance = 0.0001f;
	
	ConjugateGradient(A, simData.pressures, divergence, maxIterations, tolerance);

	CudaCalculateXVelChange<<<(numXFaces + 255) / 256, 256>>>(simData);
	CudaCalculateYVelChange<<<(numYFaces + 255) / 256, 256>>>(simData);
	CudaCalculateZVelChange<<<(numZFaces + 255) / 256, 256>>>(simData);
}

#pragma endregion Project


#pragma region ConjugateGradient

__device__ bool IsValidCell(int i, int j, int k, int *gridRes)
{
	if (i >= gridRes[0] || j >= gridRes[1] || k >= gridRes[2])
		return false;

	if (i < 0 || j < 0 || k < 0)
		return false;

	return true;
}

__global__ void CudaApply(CellDataMatrix matrix, thrust::device_ptr<float> vector, thrust::device_ptr<float> result, int numCells, int *gridRes)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numCells)
	{
		int remainder;
		int k = (int)(tid / (gridRes[0] * gridRes[1]));
		remainder = tid % (gridRes[0] * gridRes[1]);

		int j = (int)(remainder / gridRes[0]);
		remainder = remainder % gridRes[0];

		int i = remainder;

		float diag = 0.0f;
		float plusI = 0.0f;
		float plusJ = 0.0f;
		float plusK = 0.0f;
		float minusI = 0.0f;
		float minusJ = 0.0f;
		float minusK = 0.0f;

		diag = matrix.diag[tid] * vector[tid];

		if (IsValidCell(i+1, j, k, gridRes)) 
			plusI = matrix.plusI[tid] * vector[tid + 1];

		if (IsValidCell(i, j+1, k, gridRes)) 
			plusJ = matrix.plusJ[tid] * vector[tid + gridRes[0]];

		if (IsValidCell(i, j, k+1, gridRes)) 
			plusK = matrix.plusK[tid] * vector[tid + (gridRes[0] * gridRes[1])];

		if (IsValidCell(i-1, j, k, gridRes)) 
			minusI = matrix.plusI[tid - 1] * vector[tid - 1];

		if (IsValidCell(i, j-1, k, gridRes)) 
			minusJ = matrix.plusJ[tid - gridRes[0]] * vector[tid - gridRes[0]];

		if (IsValidCell(i, j, k-1, gridRes)) 
			minusK = matrix.plusK[tid - (gridRes[0] * gridRes[1])] * vector[tid - (gridRes[0] * gridRes[1])];

		result[tid] = diag + plusI + plusJ + plusK + minusI + minusJ + minusK;
	}
}

struct mulPlus_functor 
{ 
	const float a; 
	mulPlus_functor(float _a) : a(_a) {} 
	__host__ __device__ float operator()(const float& x, const float& y) const 
	{ 
		return (a * x) + y; 
	} 
};

struct mulMinus_functor 
{ 
	const float a; 
	mulMinus_functor(float _a) : a(_a) {} 
	__host__ __device__ float operator()(const float& x, const float& y) const 
	{ 
		return y - (a * x); 
	} 
};

struct max_functor 
{ 
	__host__ __device__ float operator()(const float& x, const float& y) const 
	{ 
		return glm::max(glm::abs(x), glm::abs(y)); 
	} 
};

void Fluid::ConjugateGradient(CellDataMatrix &A, float *pressure, thrust::device_vector<float> &divergence, int maxIterations, float tolerance)
{
	thrust::device_vector<float> dP(numGridCells);
	thrust::fill(dP.begin(), dP.end(), 0.0f);

	thrust::device_vector<float> dR(divergence);

	thrust::device_vector<float> dZ(divergence);
	thrust::device_vector<float> dS(divergence);

//	thrust::device_vector<float> dZ(numGridCells);
//	thrust::fill(dZ.begin(), dZ.end(), 0.0f);
//	ApplyPreconditioner(A, dR, dZ);
//	thrust::device_vector<float> dS(dZ);

	float sigma = thrust::inner_product(dZ.begin(), dZ.end(), dR.begin(), 0.0f);

	for (int iteration = 0; iteration < maxIterations; iteration++) 
	{
		double rho = sigma;

		CudaApply<<<(numGridCells + 255) / 256, 256>>>(A, &dS[0], &dZ[0], numGridCells, simData.gridRes);
		
		float hostDot = thrust::inner_product(dZ.begin(), dZ.end(), dS.begin(), 0.0f);

		float alpha = rho / hostDot;

		thrust::transform(dS.begin(), dS.end(), dP.begin(), dP.begin(), mulPlus_functor(alpha));

		thrust::transform(dZ.begin(), dZ.end(), dR.begin(), dR.begin(), mulMinus_functor(alpha));

		float max = thrust::reduce(dR.begin(), dR.end(), 0.0, max_functor());

		if (max <= tolerance)
			break;

		dZ = dR;
//		ApplyPreconditioner(A, dR, dZ);

		hostDot = thrust::inner_product(dZ.begin(), dZ.end(), dR.begin(), 0.0f);

		float sigmaNew = hostDot;

		float beta = sigmaNew / rho;

		thrust::transform(dS.begin(), dS.end(), dZ.begin(), dS.begin(), mulPlus_functor(beta));

		sigma = sigmaNew;
	}


	float *blah = thrust::raw_pointer_cast(dP.data());
	HANDLE_ERROR( cudaMemcpy(pressure, blah, numGridCells * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	return;		
}

#pragma endregion ConjugateGradient


#pragma region Temperature&Density

__global__ void CudaAdvectTemperature(SimData tempData, SimData simData)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < simData.numCells[0])
	{
		int remainder;
		int k = (int)(tid / (simData.gridRes[0] * simData.gridRes[1]));
		remainder = tid % (simData.gridRes[0] * simData.gridRes[1]);

		int j = (int)(remainder / simData.gridRes[0]);
		remainder = remainder % simData.gridRes[0];

		int i = remainder;

		glm::vec3 pos = GetCentre(i, j, k, simData.dx[0]);
		glm::vec3 vel = GetVelocity(pos, simData);
		glm::vec3 prevPos = pos - (0.5f * simData.dt[0] * vel);

		vel = GetVelocity(prevPos, simData);
		prevPos = pos - (simData.dt[0] * vel);

		float dist = Interpolate(prevPos, simData.temperatures, simData.gridRes, simData.dx[0], 3);
		tempData.temperatures[tid] = dist;
	}
}

void Fluid::AdvectTemperature()
{
	HANDLE_ERROR( cudaMemcpy(tempData.temperatures, simData.temperatures, numGridCells * sizeof(float), cudaMemcpyDeviceToDevice) );

	CudaAdvectTemperature<<<(numGridCells + 255) / 256, 256>>>(tempData, simData);

	HANDLE_ERROR( cudaMemcpy(simData.temperatures, tempData.temperatures, numGridCells * sizeof(float), cudaMemcpyDeviceToDevice) );
}


__global__ void CudaAdvectDensity(SimData tempData, SimData simData)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < simData.numCells[0])
	{
		int remainder;
		int k = (int)(tid / (simData.gridRes[0] * simData.gridRes[1]));
		remainder = tid % (simData.gridRes[0] * simData.gridRes[1]);

		int j = (int)(remainder / simData.gridRes[0]);
		remainder = remainder % simData.gridRes[0];

		int i = remainder;

		glm::vec3 pos = GetCentre(i, j, k, simData.dx[0]);
		glm::vec3 vel = GetVelocity(pos, simData);
		glm::vec3 prevPos = pos - (0.5f * simData.dt[0] * vel);

		vel = GetVelocity(prevPos, simData);
		prevPos = pos - (simData.dt[0] * vel);

		float dist = Interpolate(prevPos, simData.densities, simData.gridRes, simData.dx[0], 3);
		tempData.densities[tid] = dist;
	}
}

void Fluid::AdvectDensity()
{
	HANDLE_ERROR( cudaMemcpy(tempData.densities, simData.densities, numGridCells * sizeof(float), cudaMemcpyDeviceToDevice) );

	CudaAdvectDensity<<<(numGridCells + 255) / 256, 256>>>(tempData, simData);

	HANDLE_ERROR( cudaMemcpy(simData.densities, tempData.densities, numGridCells * sizeof(float), cudaMemcpyDeviceToDevice) );
}

#pragma endregion Temperature&Density



#pragma region PreconditionerStuff

__global__ void CudaPreconInit(CellDataMatrix matrix, thrust::device_ptr<float> precon, int numCells, int *gridRes)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numCells)
	{
		int remainder;
		int k = (int)(tid / (gridRes[0] * gridRes[1]));
		remainder = tid % (gridRes[0] * gridRes[1]);

		int j = (int)(remainder / gridRes[0]);
		remainder = remainder % gridRes[0];

		int i = remainder;

		float tau = 0.97f;
		float e;
		float pi, pj, pk, mpi, mpj, mpk;

		float *preconPtr = thrust::raw_pointer_cast(precon);

		if(IsValidCell(i, j, k, gridRes))
		{
			pi = GetDataIJK(i-1, j, k, gridRes, preconPtr, 3);
			pj = GetDataIJK(i, j-1, k, gridRes, preconPtr, 3);
			pk = GetDataIJK(i, j, k-1, gridRes, preconPtr, 3);

			mpi = GetDataIJK(i-1, j, k, gridRes, matrix.plusI, 3);
			mpj = GetDataIJK(i, j-1, k, gridRes, matrix.plusJ, 3);
			mpk = GetDataIJK(i, j, k-1, gridRes, matrix.plusK, 3);

			pi *= pi;
			pj *= pj;
			pk *= pk;

			e = GetDataIJK(i, j, k, gridRes, matrix.diag, 3); - (mpi * mpi * pi) - (mpj * mpj * pj) - (mpk * mpk * pk) - (tau * 
																						( 
																							(mpi * pi * (GetDataIJK(i-1, j, k, gridRes, matrix.plusJ, 3) + GetDataIJK(i-1, j, k, gridRes, matrix.plusK, 3))) +
																							(mpj * pj * (GetDataIJK(i, j-1, k, gridRes, matrix.plusI, 3) + GetDataIJK(i, j-1, k, gridRes, matrix.plusK, 3))) +
																							(mpk * pk * (GetDataIJK(i, j, k-1, gridRes, matrix.plusI, 3) + GetDataIJK(i, j, k-1, gridRes, matrix.plusJ, 3)))
																						)
																					);
			
			precon[tid] = 1.0f / glm::sqrt(e + 1.0e-30);
		}
	}
}

/*
__global__ void CudaPreconInit(CellDataMatrix matrix, thrust::device_ptr<float> precon, int numCells, int *gridRes)
{
	float *preconPtr = thrust::raw_pointer_cast(precon);

	for (int k=0; k<gridRes[2]; k++)
		for (int j=0; j<gridRes[1]; j++)
			for (int i=0; i<gridRes[0]; i++)
			{
				float tau = 0.97f;
				float e;
				float pi, pj, pk, mpi, mpj, mpk;

				if(IsValidCell(i, j, k, gridRes))
				{
					pi = GetDataIJK(i-1, j, k, gridRes, preconPtr, 3);
					pj = GetDataIJK(i, j-1, k, gridRes, preconPtr, 3);
					pk = GetDataIJK(i, j, k-1, gridRes, preconPtr, 3);

					int idx = i + (j * gridRes[0]) + (k * gridRes[0] * gridRes[1]);

					mpi = GetDataIJK(i-1, j, k, gridRes, matrix.plusI, 3);
					mpj = GetDataIJK(i, j-1, k, gridRes, matrix.plusJ, 3);
					mpk = GetDataIJK(i, j, k-1, gridRes, matrix.plusK, 3);

					pi *= pi;
					pj *= pj;
					pk *= pk;

					e = GetDataIJK(i, j, k, gridRes, matrix.diag, 3); - (mpi * mpi * pi) - (mpj * mpj * pj) - (mpk * mpk * pk) - (tau * 
																								( 
																									(mpi * pi * (GetDataIJK(i-1, j, k, gridRes, matrix.plusJ, 3) + GetDataIJK(i-1, j, k, gridRes, matrix.plusK, 3))) + 
																									(mpj * pj * (GetDataIJK(i, j-1, k, gridRes, matrix.plusI, 3) + GetDataIJK(i, j-1, k, gridRes, matrix.plusK, 3))) + 
																									(mpk * pk * (GetDataIJK(i, j, k-1, gridRes, matrix.plusI, 3) + GetDataIJK(i, j, k-1, gridRes, matrix.plusJ, 3)))
																								)
																							);
					
					float value = 1.0f / glm::sqrt(e + 1.0e-30);
					SetDataIJK(i, j, k, gridRes, preconPtr, value, 3);
				}
			}
}
*/
__global__ void CudaCalcQ(CellDataMatrix matrix, thrust::device_ptr<float> precon, thrust::device_ptr<float> vector, thrust::device_ptr<float> q, int numCells, int *gridRes)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numCells)
	{
		int remainder;
		int k = (int)(tid / (gridRes[0] * gridRes[1]));
		remainder = tid % (gridRes[0] * gridRes[1]);

		int j = (int)(remainder / gridRes[0]);
		remainder = remainder % gridRes[0];

		int i = remainder;

		float t;

		if (IsValidCell(i, j, k, gridRes))
		{
			t = vector[tid]   - matrix.plusI[tid - 1] * precon[tid - 1] * q[tid - 1]
							  - matrix.plusJ[tid - gridRes[0]] * precon[tid - gridRes[0]] * q[tid - gridRes[0]]
							  - matrix.plusK[tid - (gridRes[0] * gridRes[1])] * precon[tid - (gridRes[0] * gridRes[1])] * q[tid - (gridRes[0] * gridRes[1])];

			q[tid] = t * precon[tid];
		}
	}
}

__global__ void CudaPreconFinal(CellDataMatrix matrix, thrust::device_ptr<float> precon, thrust::device_ptr<float> result, thrust::device_ptr<float> q, int numCells, int *gridRes)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numCells)
	{
		int remainder;
		int k = (int)(tid / (gridRes[0] * gridRes[1]));
		remainder = tid % (gridRes[0] * gridRes[1]);

		int j = (int)(remainder / gridRes[0]);
		remainder = remainder % gridRes[0];

		int i = remainder;

		float t;

		if (IsValidCell(i, j, k, gridRes))
		{
			float pijk = precon[tid];

			t = q[tid]   - matrix.plusI[tid] * pijk * result[tid + 1]
						 - matrix.plusJ[tid] * pijk * result[tid + gridRes[0]]
						 - matrix.plusK[tid] * pijk * result[tid + (gridRes[0] * gridRes[1])];

			result[tid] = t * pijk;
		}
	}
}

void Fluid::ApplyPreconditioner(CellDataMatrix &A, thrust::device_vector<float> &vector, thrust::device_vector<float> &result)
{
	thrust::device_vector<float> precon(numGridCells);
	thrust::fill(precon.begin(), precon.end(), 0.0f);

	CudaPreconInit<<<1, 1>>>(A, &precon[0], numGridCells, simData.gridRes);		// Needs to be consecutive

	thrust::device_vector<float> q(numGridCells);
	thrust::fill(q.begin(), q.end(), 0.0f);

	CudaCalcQ<<<(numGridCells + 255) / 256, 256>>>(A, &precon[0], &vector[0], &q[0], numGridCells, simData.gridRes);
	
	CudaPreconFinal<<<(numGridCells + 255) / 256, 256>>>(A, &precon[0], &result[0], &q[0], numGridCells, simData.gridRes);
}

#pragma endregion PreconditionerStuff

