#include "TransferFunction.h"

void TransferFunction::LoadXML(const char *filename)
{
	tinyxml2::XMLDocument doc;
	auto r = doc.LoadFile("nucleon.tfi");

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

		colours.push_back(glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f));
		opacities.push_back(a / 255.0f);

		std::cout << "intensity=" << intensity;
		std::cout << "\tcolorL r=" << r << " g=" << g << " b=" << b << " a=" << a;
		std::cout << std::endl;

		key = key->NextSiblingElement();
	}
}