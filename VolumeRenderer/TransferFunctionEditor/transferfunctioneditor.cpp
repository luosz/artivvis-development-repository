#include "transferfunctioneditor.h"
#include "ui_transferfunctioneditor.h"

TransferFunctionEditor::TransferFunctionEditor(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::TransferFunctionEditor),
intensity_histogram_view("Intensity Histogram"),
visibility_histogram_view("Visibility Histogram")
{
	ui->setupUi(this);

	// add transfer function widget
	ui->verticalLayout->addWidget(&tfView);

	// add histogram widget
	ui->verticalLayout_2->addWidget(&intensity_histogram_view);
	ui->verticalLayout_3->addWidget(&visibility_histogram_view);
	ui->verticalLayout_4->addWidget(&frustum_histogram_view);
	ui->verticalLayout_5->addWidget(&gradient_view);

	// load default transfer function
	//	filename = "../../transferfuncs/nucleon.tfi";
	filename = "../../transferfuncs/CT-Knee_spectrum_16_balance.tfi";
	//	filename = "../../transferfuncs/00.tfi";

	QByteArray array = filename.toLocal8Bit();
	char* buffer = array.data();
	openTransferFunctionFromVoreenXML(buffer);
	tfView.setTransferFunction(numIntensities, colors, intensities);

	// set up histogram
	int n = 16;
	for (int i = 0; i <= n;i++)
	{
		float intensity = i / (float)n;
		intensity_histogram_view.intensity_list.push_back(intensity);
		intensity_histogram_view.frequency_list.push_back(intensity*intensity);
		visibility_histogram_view.intensity_list.push_back(intensity);
		visibility_histogram_view.frequency_list.push_back(intensity*intensity);
	}
}

TransferFunctionEditor::~TransferFunctionEditor()
{
	delete ui;
}

void TransferFunctionEditor::on_action_Open_Transfer_Function_triggered()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), "../../transferfuncs/nucleon.tfi", tr("Voreen Transfer Function (*.tfi) ;; All (*.*)"));
	std::cout << "size" << filename.size() << std::endl;
	if (filename.size() > 0)
	{
		QByteArray array = filename.toLocal8Bit();
		char* buffer = array.data();
		std::cout << "Open transfer function from " << buffer << std::endl;
		intensities.clear();
		colors.clear();
		openTransferFunctionFromVoreenXML(buffer);
		tfView.setTransferFunction(numIntensities, colors, intensities);
	}
}

void TransferFunctionEditor::on_action_Save_Transfer_Function_triggered()
{
	tfView.getTransferFunction(numIntensities, colors, intensities);
	QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "../../transferfuncs/save_as.tfi", tr("Voreen Transfer Function (*.tfi) ;; All (*.*)"));
	std::cout << "size" << filename.size() << std::endl;
	if (filename.size() > 0)
	{
		QByteArray array = filename.toLocal8Bit();
		char* buffer = array.data();
		std::cout << "Save transfer function to " << buffer << std::endl;
		saveTransferFunctionToVoreenXML(buffer);
	}
}

void TransferFunctionEditor::on_distributeHorizontallyButton_clicked()
{
	tfView.distrubuteHorizontally();
}

void TransferFunctionEditor::on_distributeVerticallyButton_clicked()
{
	tfView.distributeVertically();
}

void TransferFunctionEditor::on_diagonalButton_clicked()
{
	tfView.makeDiagonal();
}

void TransferFunctionEditor::on_peaksButton_clicked()
{
	std::cout << "on_peaksButton_clicked\n";
}

void TransferFunctionEditor::on_rampButton_clicked()
{
	tfView.makeRamp(ui->doubleSpinBox->value());
}

void TransferFunctionEditor::on_entropyButton_clicked()
{
#ifndef NOT_USED_BY_VOLUME_RENDERER
	if (tfView.transferFunction())
	{
		std::cout<<"on_entropyButton_clicked\n";

		tfView.updateTransferFunctionFromView();
		tfView.optimizer()->numIterations = ui->spinBox->value();
		tfView.optimizer()->BalanceEdges();
		tfView.transferFunction()->LoadLookup(tfView.transferFunction()->currentColorTable);

		tfView.updateViewFromTransferFunction();
	}
#endif // NOT_USED_BY_VOLUME_RENDERER
}

void TransferFunctionEditor::on_visibilityButton_clicked()
{
#ifndef NOT_USED_BY_VOLUME_RENDERER
	if (tfView.transferFunction())
	{
		std::cout<<"on_visibilityButton_clicked\n";

		tfView.updateTransferFunctionFromView();
		tfView.optimizer()->numIterations = ui->spinBox->value();
		tfView.optimizer()->BalanceVisibility();
		tfView.transferFunction()->LoadLookup(tfView.transferFunction()->currentColorTable);

		tfView.updateViewFromTransferFunction();
	}
#endif // NOT_USED_BY_VOLUME_RENDERER
}

void TransferFunctionEditor::on_checkBox_clicked()
{
	ui->checkBox_2->setChecked(false);
	tfView.is_ma_optimizer_enable = ui->checkBox->isChecked();
	tfView.is_luo_optimizer_enable = false;
	std::cout << "Ma's optimizer " << (tfView.isMaOptimizerEnable() ? "enabled" : "disabled") << std::endl;
}

void TransferFunctionEditor::on_checkBox_2_clicked()
{
	ui->checkBox->setChecked(false);
	tfView.is_luo_optimizer_enable = ui->checkBox_2->isChecked();
	tfView.is_ma_optimizer_enable = false;
	std::cout << "Luo's optimizer " << (tfView.isLuoOptimizerEnable() ? "enabled" : "disabled") << std::endl;
}

void TransferFunctionEditor::on_flatButton_clicked()
{
	tfView.makeFlat(ui->doubleSpinBox->value());
}

void TransferFunctionEditor::on_visibilityHistogramButton_clicked()
{
#ifndef NOT_USED_BY_VOLUME_RENDERER
	VisibilityHistogram &visibilityHistogram = tfView.volumeRenderer()->renderer->visibilityHistogram;
	visibility_histogram_view.setVisibilityHistogram(visibilityHistogram.visibilities, visibilityHistogram.numVis);
	visibility_histogram_view.draw();
#endif // NOT_USED_BY_VOLUME_RENDERER
}

void TransferFunctionEditor::on_computeDistanceButton_clicked()
{
#ifndef NOT_USED_BY_VOLUME_RENDERER
	VisibilityHistogram &visibilityHistogram = tfView.volumeRenderer()->renderer->visibilityHistogram;
	//for (int i=0; i<visibilityHistogram.visibilities.size(); i++)
	//{
	//	std::cout<<i<<" "<<visibilityHistogram.visibilities[i]<<" "<<visibilityHistogram.numVis[i]<<" "<<visibilityHistogram.intensity_histogram[i]<<std::endl;
	//}
	//frustum_histogram_view.setHistogram(visibilityHistogram.visibilities, visibilityHistogram.numVis);
	//frustum_histogram_view.draw();

	std::vector<float> weights;
	weights.resize(tfView.intensities.size());
	auto size = visibilityHistogram.visibilities.size();
	float sum = 0;
	for (int i=0; i<tfView.intensities.size(); i++)
	{
		weights[i] = 0;
		for (int j=0; j<size; j++)
		{
			weights[i] += abs(tfView.intensities[i] - (j / (float)size)) * ( - visibilityHistogram.visibilities[j] * visibilityHistogram.numVis[j]);
		}
		sum += weights[i];
	}
	for (int i=0; i<tfView.intensities.size(); i++)
	{
		weights[i] = weights[i] / sum;
	}

	frustum_histogram_view.setHistogram(weights);
	frustum_histogram_view.draw();
#endif // NOT_USED_BY_VOLUME_RENDERER
}
