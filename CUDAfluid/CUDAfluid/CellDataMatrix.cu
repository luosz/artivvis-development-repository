#include "CellDataMatrix.h"

__global__ void CudaAddNeighbours(CellDataMatrix matrix, int numCells, int xRes, int yRes, int zRes)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numCells)
	{
		int remainder;
		int Z = (int)(tid / (xRes * yRes));
		remainder = tid % (xRes * yRes);

		int Y = (int)(remainder / xRes);
		remainder = remainder % xRes;

		int X = remainder;

		int numFluidNeighbors = 0;

		if (X-1 >= 0) 
		{
			matrix.plusI[tid - 1] = -1.0f;
			numFluidNeighbors++;
		}
		if (X+1 < xRes) 
		{
			matrix.plusI[tid] = -1.0f;
			numFluidNeighbors++;
		}
		if (Y-1 >= 0) 
		{
			matrix.plusJ[tid - xRes] = -1.0f;
			numFluidNeighbors++;
		}
		if (Y+1 < yRes) 
		{
			matrix.plusJ[tid] = -1.0f;
			numFluidNeighbors++;
		}
		if (Z-1 >= 0) 
		{
			matrix.plusK[tid - (xRes*yRes)] = -1.0f;
			numFluidNeighbors++;
		}
		if (Z+1 < zRes) 
		{
			matrix.plusK[tid] = -1.0f;
			numFluidNeighbors++;
		}

		// Set the diagonal:
		matrix.diag[tid] = numFluidNeighbors;
	}
}

void CellDataMatrix::Init()
{
	HANDLE_ERROR( cudaMalloc((void**)&plusI, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&plusJ, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&plusK, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&diag, numGridCells * sizeof(float)) );

	HANDLE_ERROR( cudaMemset(plusI, 0, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(plusJ, 0, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(plusK, 0, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(diag, 0, numGridCells * sizeof(float)) );

	cudaDeviceSynchronize();

	CudaAddNeighbours<<<(numGridCells + 255) / 256, 256>>>(*this, numGridCells, gridXRes, gridYRes, gridZRes);
}
