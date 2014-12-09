#ifndef TRANSFERFUNCTIONEDITOR_H
#define TRANSFERFUNCTIONEDITOR_H

#include <QMainWindow>
#include <iostream>
#include <QFileDialog>

#include "import_volume_renderer.h"

#include "glm/glm.hpp"
#include "tinyxml2.h"
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
		auto r = doc.LoadFile(filename);

		if (r != tinyxml2::XML_NO_ERROR)
		{
			std::cout << "failed to open file " << filename << std::endl;
			return;
		}

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

	/// save transfer function as Voreen XML
	void saveTransferFunctionToVoreenXML(const char *filename)
	{
		tinyxml2::XMLDocument doc;

		auto declaration = doc.NewDeclaration();
		doc.InsertEndChild(declaration);
		auto voreenData = doc.NewElement("VoreenData");
		voreenData->SetAttribute("version", 1);
		auto transFuncIntensity = doc.NewElement("TransFuncIntensity");
		transFuncIntensity->SetAttribute("type", "TransFuncIntensity");

		// add domain
		auto domain = doc.NewElement("domain");
		domain->SetAttribute("x", 0);
		domain->SetAttribute("y", 1);
		transFuncIntensity->InsertEndChild(domain);

		// add threshold
		auto threshold = doc.NewElement("threshold");
		threshold->SetAttribute("x", 0);
		threshold->SetAttribute("y", 1);
		transFuncIntensity->InsertEndChild(threshold);

		// add Keys
		auto size = intensities.size();
		auto keys = doc.NewElement("Keys");
		for (int i = 0; i < size; i++)
		{
			auto key = doc.NewElement("key");
			key->SetAttribute("type", "TransFuncMappingKey");
			auto intensity = doc.NewElement("intensity");
			intensity->SetAttribute("value", intensities[i]);
			auto split = doc.NewElement("split");
			split->SetAttribute("value", "false");
			auto colorL = doc.NewElement("colorL");
			auto c = colors[i];
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

		auto r = doc.SaveFile(filename);
		if (r != tinyxml2::XML_NO_ERROR)
		{
			std::cout << "failed to save file " << filename << std::endl;
		}
	}

#ifndef NOT_USED_BY_VOLUME_RENDERER
	void init(VolumeRenderer &volumeRenderer)
	{
		std::cout << "TransferFunctionEditor::init" << std::endl;
		tf.transfer_function = &volumeRenderer.renderer->transferFunction;
		if (tf.transfer_function)
		{
			std::cout << "tf.transfer_function is not NULL" << std::endl;
		} 
		else
		{
			std::cout << "tf.transfer_function is NULL" << std::endl;
		}

		if (tf.transfer_function)
		{
			auto &frequencies = tf.transfer_function->intensityOptimizerV2->frequencies;
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
			//intensity_histogram.draw();
			auto &visibilities = tf.transfer_function->intensityOptimizerV2->visibilityHistogram->visibilities;
			auto &numVis = tf.transfer_function->intensityOptimizerV2->visibilityHistogram->numVis;
			visibility_histogram.intensities.clear();
			visibility_histogram.frequencies.clear();
			for (int i = 0; i < visibilities.size();i++)
			{
				visibility_histogram.intensities.push_back(i / (float)size);
				visibility_histogram.frequencies.push_back(visibilities[i]);
			}
			//visibility_histogram.draw();
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
    TransferFunctionView tf;
    int numIntensities;
    std::vector<glm::vec4> colors;
    std::vector<float> intensities;
	QString filename;
	HistogramView intensity_histogram;
	HistogramView visibility_histogram;
};

#endif // TRANSFERFUNCTIONEDITOR_H
