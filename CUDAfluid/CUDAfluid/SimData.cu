#include "SimData.h"


void SimData::Init()
{
	// Allocate Memory
	HANDLE_ERROR( cudaMalloc((void**)&gridRes, 3 * sizeof(int)) );
	HANDLE_ERROR( cudaMalloc((void**)&numCells,  sizeof(int)) );

	HANDLE_ERROR( cudaMalloc((void**)&xFaceRes, 3 * sizeof(int)) );
	HANDLE_ERROR( cudaMalloc((void**)&yFaceRes, 3 * sizeof(int)) );
	HANDLE_ERROR( cudaMalloc((void**)&zFaceRes, 3 * sizeof(int)) );

	HANDLE_ERROR( cudaMalloc((void**)&numFaces, 3 * sizeof(int)) );

	HANDLE_ERROR( cudaMalloc((void**)&dt,  sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&dx,  sizeof(float)) );

	HANDLE_ERROR( cudaMalloc((void**)&xVelocities, (gridXRes+1) * gridYRes * gridZRes * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&yVelocities, gridXRes * (gridYRes+1) * gridZRes * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&zVelocities, gridXRes * gridYRes * (gridZRes+1) * sizeof(float)) );

	HANDLE_ERROR( cudaMalloc((void**)&pressures, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&densities, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&temperatures, numGridCells * sizeof(float)) );

	int Xp1 = gridXRes + 1;
	int Yp1 = gridYRes + 1;
	int Zp1 = gridZRes + 1;

	// Initialize values
	HANDLE_ERROR( cudaMemcpy(&gridRes[0], &gridXRes, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&gridRes[1], &gridYRes, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&gridRes[2], &gridZRes, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(numCells, &numGridCells, sizeof(int), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(&xFaceRes[0], &Xp1, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&xFaceRes[1], &gridYRes, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&xFaceRes[2], &gridZRes, sizeof(int), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(&yFaceRes[0], &gridXRes, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&yFaceRes[1], &Yp1, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&yFaceRes[2], &gridZRes, sizeof(int), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(&zFaceRes[0], &gridXRes, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&zFaceRes[1], &gridYRes, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&zFaceRes[2], &Zp1, sizeof(int), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(&numFaces[0], &numXFaces, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&numFaces[1], &numYFaces, sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(&numFaces[2], &numZFaces, sizeof(int), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(dt, &timestep, sizeof(float), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dx, &cellSize, sizeof(float), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemset(xVelocities, 0, numXFaces * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(yVelocities, 0, numYFaces * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(zVelocities, 0, numZFaces * sizeof(float)) );

	HANDLE_ERROR( cudaMemset(pressures, 0, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(densities, 0, numGridCells * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(temperatures, 0, numGridCells * sizeof(float)) );

	cudaDeviceSynchronize();
}