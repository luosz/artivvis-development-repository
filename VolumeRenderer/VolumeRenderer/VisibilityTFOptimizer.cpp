#include "VisibilityTFOptimizer.h"

void VisibilityTFOptimizer::Init()
{
	Es.resize(256);
	Ev.resize(256);
	Ec.resize(256);
	energyFunc.resize(256);

	iterations = 0;
}


void VisibilityTFOptimizer::Optimize(VolumeDataset &volume, VisibilityHistogram &visibilityHistogram, TransferFunction &transferFunction)
{
	
	float prevEnergy = 10000000.0f;

//	if (iterations == 0)
//	{
//		for (int i=0; i<visibilityHistogram.numBins; i++)
//		{
//			transferFunction.currentColorTable[i].a = 0.0f;
//		}
//	}

//	while (iterations < 1)
//	{
		// Fits nicely because 1D transfer function is divided in 256 bins anyway, must change if different amount of bins
		for (int i=0; i<visibilityHistogram.numBins; i++)
		{
			Es[i] = glm::pow((transferFunction.currentColorTable[i].a - transferFunction.origColorTable[i].a), 2.0f);
		}


		for (int i=0; i<visibilityHistogram.numBins; i++)
		{
			Ev[i] = -(transferFunction.origColorTable[i].a * visibilityHistogram.visibilities[i]);
		}

		float min = 0.0f;
		float max = 1.0f;
		
		for (int i=0; i<transferFunction.numIntensities; i++)
		{
			Ec[i] = (glm::pow(glm::max((min - transferFunction.colors[i].a), 0.0f), 2.0f) + glm::pow(glm::max((transferFunction.colors[i].a - max), 0.0f), 2.0f));
		}

		float beta1 = 0.5f;
		float beta2 = 0.5f;
		float beta3 = 1.0f;
		float energy = 0.0f;

		for (int i=0; i<visibilityHistogram.numBins; i++)
		{
			energyFunc[i] = (beta1 * Es[i]);// + (beta2 * Ev[i]);		//  + (beta3 * Ec[i])
			energy += (beta1 * Es[i]);// + (beta2 * Ev[i]);  // 
		}

		float stepsize = 0.1f;

		for (int i=0; i<visibilityHistogram.numBins; i++)
		{
			float gradient = beta1 * ((2.0f * transferFunction.currentColorTable[i].a) - (2.0f * transferFunction.origColorTable[i].a));
//			float gradient = beta2 * transferFunction.origColorTable[i].a / transferFunction.currentColorTable[i].a;

			transferFunction.currentColorTable[i].a -= stepsize * gradient;

			transferFunction.currentColorTable[i].a = glm::clamp(transferFunction.currentColorTable[i].a, 0.0f, 1.0f);
		}

			std::cout << iterations << ": " << energy << std::endl;


		iterations++;
//	}

	transferFunction.CopyToTex(transferFunction.currentColorTable);
}

// + (beta2 * transferFunction.origColorTable[i].a * visibilityHistogram.visibilities[i] * glm::exp(-visibilityHistogram.visibilities[i]));


void VisibilityTFOptimizer::DrawEnergy(ShaderManager shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(SimpleShader);

	int uniformLoc;

	glm::mat4 model_mat = glm::mat4(1.0f);

	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glm::mat4 tempView = glm::lookAt(glm::vec3(0.5f, 0.5f, 2.0f), glm::vec3(0.5f, 0.5f, 0.0f), glm::vec3(0.0f,1.0f,0.0f));
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &tempView[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);


	glBegin(GL_LINES);

	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);

	glColor3f(0.0f, 0.0f, 1.0f);
	for (int i=0; i<256; i++)
	{
		glVertex3f(i / 255.0f, 0.0f, 0.0f);
		glVertex3f(i / 255.0f, energyFunc[i], 0.0f);
	}

	glEnd();
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