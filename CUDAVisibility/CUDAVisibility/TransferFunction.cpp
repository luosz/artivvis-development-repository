#include "TransferFunction.h"

void TransferFunction::Init(const char *filename, VolumeDataset &volume_)
{
	colorTable.resize(256);
	LoadXML(filename);
	LoadLookup();

//	optimizer = new IntensityTFOptimizer(volume_, numIntensities, &colors[0], &intensities[0]);
	optimizer = new VisibilityTFOptimizer(volume_, numIntensities, &colorTable[0], &intensities[0]);

	targetIntensity = 0.5f;
}

void TransferFunction::LoadXML(const char *filename)
{
	tinyxml2::XMLDocument doc;
//	auto r = doc.LoadFile("nucleon.tfi");
	auto r = doc.LoadFile("CT-Knee_spectrum_16_balance.tfi");
//	auto r = doc.LoadFile("../../Samples/downsampled vortex/90.tfi");

	if (r != tinyxml2::XML_NO_ERROR)
		std::cout << "failed to open file" << std::endl;

	auto transFuncIntensity = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity");

	auto key = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity")->FirstChildElement("Keys")->FirstChildElement("key");

	while (key)
	{
		float intensity = atof(key->FirstChildElement("intensity")->Attribute("value"));
		intensities.push_back(intensity);

		int r = atoi(key->FirstChildElement("colorL")->Attribute("r"));
		int g = atoi(key->FirstChildElement("colorL")->Attribute("g"));
		int b = atoi(key->FirstChildElement("colorL")->Attribute("b"));
		int a = atoi(key->FirstChildElement("colorL")->Attribute("a"));

		colors.push_back(glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));

		std::cout << "intensity=" << intensity;
		std::cout << "\tcolorL r=" << r << " g=" << g << " b=" << b << " a=" << a;
		std::cout << std::endl;

		key = key->NextSiblingElement();
	}

	numIntensities = intensities.size();
}

void TransferFunction::LoadLookup()
{
	glm::vec4 previousColor(0.0f);
	float previousIntensity = 0.0f;
	int next = 0;

	for (int i=0; i<256; i++)
	{
		float currentIntensity = (float)i / (float)255;

		while (next < numIntensities && currentIntensity > intensities[next])
		{
			previousIntensity = intensities[next];
			previousColor = colors[next];
			next++;
		}

		if (next < numIntensities)
			colorTable[i] = LERPColor(previousColor, colors[next], previousIntensity, intensities[next], currentIntensity);
		else
			colorTable[i] = LERPColor(previousColor, glm::vec4(0.0f), previousIntensity, 1.0f, currentIntensity);
	}

	glGenTextures(1, &tfTexture);
    glBindTexture(GL_TEXTURE_1D, tfTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 256, 0, GL_RGBA, GL_FLOAT, &colorTable[0]); 
	glBindTexture(GL_TEXTURE_1D, 0);
}

glm::vec4 TransferFunction::LERPColor(glm::vec4 firstColor, glm::vec4 secondColor, float firstIntensity, float secondIntensity, float currentIntensity)
{
	float fraction = (currentIntensity - firstIntensity) / (secondIntensity - firstIntensity);

	return firstColor + ((secondColor - firstColor) * fraction);
}


void TransferFunction::Optimize(VisibilityHistogram &histogram)
{
	optimizer->Optimize(targetIntensity, histogram);
	LoadLookup();
}
