#include "VisibilityTFOptimizer.h"

void VisibilityTFOptimizer::Init()
{
	Es.resize(256);
	Ev.resize(256);
	energyFunc.resize(256);
}


void VisibilityTFOptimizer::Optimize(VolumeDataset &volume, VisibilityHistogram &visibilityHistogram, TransferFunction &transferFunction)
{
	int iterations = 0;
	float prevEnergy = 10000000.0f;

//	for (int i=0; i<visibilityHistogram.numBins; i++)
//	{
//		transferFunction.currentColorTable[i].a = 0.0f;
//	}

	while (iterations < 5000)
	{
		// Fits nicely because 1D transfer function is divided in 256 bins anyway, must change if different amount of bins
		for (int i=0; i<visibilityHistogram.numBins; i++)
		{
			Es[i] = glm::pow((transferFunction.currentColorTable[i].a - transferFunction.origColorTable[i].a), 2.0f);
		}


		for (int i=0; i<visibilityHistogram.numBins; i++)
		{
			Ev[i] = -(transferFunction.origColorTable[i].a * visibilityHistogram.visibilities[i]);
		}

		float beta1 = 0.5f;
		float beta2 = 0.5f;
		float beta3 = 1.0f;
		float energy = 0.0f;

		for (int i=0; i<visibilityHistogram.numBins; i++)
		{
			energyFunc[i] = (beta1 * Es[i]) + (beta2 * Ev[i]);
			energy += (beta1 * Es[i]) + (beta2 * Ev[i]);
		}

		float stepsize = 0.1f;

		for (int i=1; i<visibilityHistogram.numBins - 1; i++)
		{
//			float gradient = (energyFunc[i + 1] - energyFunc[i - 1]);
//
//			if (gradient > 0.0f)
//				transferFunction.currentColorTable[i].a -= stepsize;
//			else if (gradient < 0.0f)
//				transferFunction.currentColorTable[i].a += stepsize;

			

			transferFunction.currentColorTable[i].a -= stepsize * (energyFunc[i + 1] - energyFunc[i - 1]);

			transferFunction.currentColorTable[i].a = glm::clamp(transferFunction.currentColorTable[i].a, 0.0f, 1.0f);
		}

		if (iterations % 10 == 0)
			std::cout << iterations << ": " << energy << std::endl;

		if (energy < 0.1f)
			break;
//		else
//			prevEnergy = energy;
//
		iterations++;
	}

	transferFunction.CopyToTex(transferFunction.currentColorTable);
}






/*
float Es = 0.0f;

	// Fits nicely because 1D transfer function is divided in 256 bins anyway, must change if different amount of bins
	for (int i=0; i<visibilityHistogram.numBins; i++)
	{
		Es += glm::pow((transferFunction.currentColorTable[i].a - transferFunction.origColorTable[i].a), 2.0f);
	}


	float Ev = 0.0f;

	for (int i=0; i<visibilityHistogram.numBins; i++)
	{
		Ev -= (transferFunction.origColorTable[i].a * visibilityHistogram.visibilities[i]);
	}


	float Ec = 0.0f;

	float min = 0.0f;
	float max = 1.0f;

	for (int i=0; i<transferFunction.numIntensities; i++)
	{
		Ec += (glm::pow(glm::max((min - transferFunction.colors[i].a), 0.0f), 2.0f) + glm::pow(glm::max((transferFunction.colors[i].a - max), 0.0f), 2.0f));
	}


	float beta1 = 0.5f;
	float beta2 = 0.5f;
	float beta3 = 1.0f;

	float energy = (beta1 * Es) + (beta2 * Ev) + (beta3 * Ec);
	*/