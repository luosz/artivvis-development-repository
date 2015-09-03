#include "VisibilityTFOptimizer.h"

VisibilityTFOptimizer::VisibilityTFOptimizer(VolumeDataset *volume_, VisibilityHistogram *visibilityHistogram_, TransferFunction *transferFunction_)
{
	volume = volume_;
	visibilityHistogram = visibilityHistogram_;
	transferFunction = transferFunction_;

	// Fits nicely because 1D transfer function is divided in 256 bins anyway, must change if different amount of bins
	numBins = 256;

	// Resize to match number of bins
	Es.resize(numBins);
	Ev.resize(numBins);
	Ec.resize(numBins);
	energyFunc.resize(numBins);

	iterations = 0;
}


void VisibilityTFOptimizer::Optimize()
{
	// Ma's energy function for user satisfaction - square distance between current transfer function and reference transfer function
	for (int i=0; i<numBins; i++)
	{
		Es[i] = glm::pow((transferFunction->currentColorTable[i].a - transferFunction->origColorTable[i].a), 2.0f);
	}

	// Ma's energy function for visibility
//	for (int i=0; i<numBins; i++)
//	{
//		Ev[i] = -(transferFunction.origColorTable[i].a * visibilityHistogram.visibilities[i]);
//	}

	// My energy function for visibility - square distance between current transfer function and visibility histogram
	for (int i=0; i<numBins; i++)
	{
		Ev[i] = glm::pow((visibilityHistogram->visibilities[i] - transferFunction->currentColorTable[i].a), 2.0f);
	}

	float min = 0.0f;
	float max = 1.0f;
	
	// Clamping energy for specific regions, I don't use it and instead just clamp opacity between 0 and 1 after optimization
	for (int i=0; i<transferFunction->numIntensities; i++)
	{
		Ec[i] = (glm::pow(glm::max((min - transferFunction->colors[i].a), 0.0f), 2.0f) + glm::pow(glm::max((transferFunction->colors[i].a - max), 0.0f), 2.0f));
	}



	// Weights of the different energy components
	float beta1 = 0.05f;
	float beta2 = 0.95f;
	float beta3 = 0.0f;
	float energy = 0.0f;

	// Accumulate for visualising on graph and observing overall energy drop
	for (int i=0; i<numBins; i++)
	{
		energyFunc[i] = (beta1 * Es[i]) + (beta2 * Ev[i]) + (beta3 * Ec[i]);
		energy += (beta1 * Es[i]) + (beta2 * Ev[i]) + (beta3 * Ec[i]);
	}




	float stepsize = 0.2f;

	for (int i=0; i<numBins; i++)
	{
		// Gradient is calculated by differentiating various energy components with respect to current opacity function
		float gradient = (beta1 * ((2.0f * transferFunction->currentColorTable[i].a) - (2.0f * transferFunction->origColorTable[i].a))) + (beta2 * ((2.0f * transferFunction->currentColorTable[i].a) - (2.0f * visibilityHistogram->visibilities[i])));

//		float gradient = (beta1 * ((2.0f * transferFunction.currentColorTable[i].a) - (2.0f * transferFunction.origColorTable[i].a)));

		// Gradient descent to minimize energy function
		transferFunction->currentColorTable[i].a -= stepsize * gradient;

		// Clamp final values
		transferFunction->currentColorTable[i].a = glm::clamp(transferFunction->currentColorTable[i].a, 0.0f, 1.0f);
	}

	// Print energy
//	std::cout << iterations << ": " << energy << std::endl;

	iterations++;

	// Copy updated opacity function to transfer function texture
	transferFunction->CopyToTex(transferFunction->currentColorTable);
}


// Draw Energy
void VisibilityTFOptimizer::Draw(ShaderManager &shaderManager, Camera &camera)
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

	// Draw graph axes
	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);

	// Draw user satisfaction energy
//	glColor3f(0.0f, 0.0f, 1.0f);
//	for (int i=0; i<256; i++)
//	{
//		glVertex3f(i / 255.0f, 0.0f, 0.0f);
//		glVertex3f(i / 255.0f, Es[i]*5.0f, 0.0f);
//	}

	// Draw visibility energy
	glColor3f(0.0f, 1.0f, 0.0f);
	for (int i=0; i<256; i++)
	{
		glVertex3f(i / 255.0f, 0.0f, 0.0f);
		glVertex3f(i / 255.0f, -energyFunc[i]*2.0f, 0.0f);
	}

	glEnd();
}