#ifndef TRANSFERFUNCTIONEDITOR_H
#define TRANSFERFUNCTIONEDITOR_H

#include <QMainWindow>
#include <iostream>
#include <QFileDialog>

#include "import_volume_renderer.h"

#include "glm/glm.hpp"
#include "../gl/include/tinyxml2.h"
#include "graphwidget.h"
#include "ControlEdge.h"
#include "ControlPoint.h"
#include "TransferFunctionView.h"
#include "HistogramView.h"

namespace Ui {
class TransferFunctionEditor;
}

class TransferFunctionEditor : public QMainWindow
{
    Q_OBJECT

public:
    explicit TransferFunctionEditor(QWidget *parent = 0);
    ~TransferFunctionEditor();

	/// open transfer function from Voreen XML
    void openTransferFunctionFromVoreenXML(const char *filename)
    {
		tinyxml2::XMLDocument doc;
		tinyxml2::XMLError r = doc.LoadFile(filename);

		if (r != tinyxml2::XML_NO_ERROR)
		{
			std::cout << "failed to open file " << filename << std::endl;
			return;
		}

        tinyxml2::XMLElement* transFuncIntensity = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity");

        tinyxml2::XMLElement* key = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity")->FirstChildElement("Keys")->FirstChildElement("key");

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

	/// save transfer function as Voreen XML
	void saveTransferFunctionToVoreenXML(const char *filename)
	{
		tinyxml2::XMLDocument doc;

		tinyxml2::XMLDeclaration* declaration = doc.NewDeclaration();
		doc.InsertEndChild(declaration);
		tinyxml2::XMLElement* voreenData = doc.NewElement("VoreenData");
		voreenData->SetAttribute("version", 1);
		tinyxml2::XMLElement* transFuncIntensity = doc.NewElement("TransFuncIntensity");
		transFuncIntensity->SetAttribute("type", "TransFuncIntensity");

		// add domain
		tinyxml2::XMLElement* domain = doc.NewElement("domain");
		domain->SetAttribute("x", 0);
		domain->SetAttribute("y", 1);
		transFuncIntensity->InsertEndChild(domain);

		// add threshold
		tinyxml2::XMLElement* threshold = doc.NewElement("threshold");
		threshold->SetAttribute("x", 0);
		threshold->SetAttribute("y", 1);
		transFuncIntensity->InsertEndChild(threshold);

		// add Keys
		int size = intensities.size();
		tinyxml2::XMLElement* keys = doc.NewElement("Keys");
		for (int i = 0; i < size; i++)
		{
			tinyxml2::XMLElement* key = doc.NewElement("key");
			key->SetAttribute("type", "TransFuncMappingKey");
			tinyxml2::XMLElement* intensity = doc.NewElement("intensity");
			intensity->SetAttribute("value", intensities[i]);
			tinyxml2::XMLElement* split = doc.NewElement("split");
			split->SetAttribute("value", "false");
			tinyxml2::XMLElement* colorL = doc.NewElement("colorL");
			glm::vec4 c = colors[i];
			colorL->SetAttribute("r", static_cast<int>(c.r * 255));
			colorL->SetAttribute("g", static_cast<int>(c.g * 255));
			colorL->SetAttribute("b", static_cast<int>(c.b * 255));
			colorL->SetAttribute("a", static_cast<int>(c.a * 255));
			key->InsertEndChild(intensity);
			key->InsertEndChild(split);
			key->InsertEndChild(colorL);
			keys->InsertEndChild(key);
		}
		transFuncIntensity->InsertEndChild(keys);

		voreenData->InsertEndChild(transFuncIntensity);
		doc.InsertEndChild(voreenData);

		tinyxml2::XMLError r = doc.SaveFile(filename);
		if (r != tinyxml2::XML_NO_ERROR)
		{
			std::cout << "failed to save file " << filename << std::endl;
		}
	}

#ifndef NOT_USED_BY_VOLUME_RENDERER
	TransferFunction *transferFunction()
	{
		return tfView.transfer_function;
	}

	IntensityTFOptimizerV2 *intensityTFOptimizerV2()
	{
		return tfView.intensityTFOptimizerV2();
	}

	void init(VolumeRenderer &volumeRenderer)
	{
		std::cout << "TransferFunctionEditor::init" << std::endl;
		tfView.transfer_function = &volumeRenderer.renderer->transferFunction;
		tfView.renderer = volumeRenderer.renderer;
		//if (transferFunction())
		//{
		//	std::cout << "tfView.transfer_function is not NULL" << std::endl;
		//} 
		//else
		//{
		//	std::cout << "tfView.transfer_function is NULL" << std::endl;
		//}

		if (transferFunction())
		{
			//auto &frequencies = tfView.transfer_function->intensityOptimizerV2->frequencies;
			auto &frequencies = intensityTFOptimizerV2()->frequencies;
			//std::cout << "frequencies size " << frequencies.size() << std::endl;
			auto size = frequencies.size();
			float max = 0;
			for (auto i=frequencies.begin(); i!=frequencies.end(); i++)
			{
				max = std::max((float)*i, max);
			}
			intensity_histogram.intensities.clear();
			intensity_histogram.frequencies.clear();
			for (int i = 0; i < size; i++)
			{
				intensity_histogram.intensities.push_back(i / (float)size);
				intensity_histogram.frequencies.push_back(frequencies[i]/max);
				//std::cout<<"frequencies "<<i<<" "<<frequencies[i]<<std::endl;
			}
			intensity_histogram.draw();
			//auto &visibilities = tfView.transfer_function->intensityOptimizerV2->visibilityHistogram->visibilities;
			//auto &numVis = tfView.transfer_function->intensityOptimizerV2->visibilityHistogram->numVis;
			auto &visibilities = intensityTFOptimizerV2()->visibilityHistogram->visibilities;
			auto &numVis = intensityTFOptimizerV2()->visibilityHistogram->numVis;
			visibility_histogram.intensities.clear();
			visibility_histogram.frequencies.clear();
			for (int i = 0; i < visibilities.size();i++)
			{
				visibility_histogram.intensities.push_back(i / (float)size);
				visibility_histogram.frequencies.push_back(visibilities[i]);
			}
			visibility_histogram.draw();
		}
	}
#endif // NOT_USED_BY_VOLUME_RENDERER

private slots:
    void on_action_Open_Transfer_Function_triggered();

    void on_action_Save_Transfer_Function_triggered();

    void on_distributeHorizontallyButton_clicked();

    void on_distributeVerticallyButton_clicked();

    void on_diagonalButton_clicked();
    void on_peaksButton_clicked();

    void on_rampButton_clicked();

    void on_entropyButton_clicked();

    void on_visibilityButton_clicked();

    void on_checkBox_clicked();

    void on_checkBox_2_clicked();

    void on_flatButton_clicked();

private:
    Ui::TransferFunctionEditor *ui;

public:
    TransferFunctionView tfView;
    int numIntensities;
    std::vector<glm::vec4> colors;
    std::vector<float> intensities;
	QString filename;
	HistogramView intensity_histogram;
	HistogramView visibility_histogram;
};

#endif // TRANSFERFUNCTIONEDITOR_H
