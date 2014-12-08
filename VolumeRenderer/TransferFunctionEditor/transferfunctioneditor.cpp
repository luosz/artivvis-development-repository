#include "transferfunctioneditor.h"
#include "ui_transferfunctioneditor.h"

TransferFunctionEditor::TransferFunctionEditor(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::TransferFunctionEditor)
{
	ui->setupUi(this);

	// add transfer function widget
	ui->verticalLayout->addWidget(&tf);

	// load default transfer function
	//	filename = "../../transferfuncs/nucleon.tfi";
	filename = "../../transferfuncs/CT-Knee_spectrum_16_balance.tfi";
	//	filename = "../../transferfuncs/00.tfi";

	QByteArray array = filename.toLocal8Bit();
	char* buffer = array.data();
	openTransferFunctionFromVoreenXML(buffer);
	tf.setTransferFunction(numIntensities, colors, intensities);

	tf.is_ma_optimizer_enable = ui->checkBox->isChecked();
	tf.is_luo_optimizer_enable = ui->checkBox_2->isChecked();
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
		tf.setTransferFunction(numIntensities, colors, intensities);
	}
}

void TransferFunctionEditor::on_action_Save_Transfer_Function_triggered()
{
	tf.getTransferFunction(numIntensities, colors, intensities);
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
	tf.distrubuteHorizontally();
}

void TransferFunctionEditor::on_distributeVerticallyButton_clicked()
{
	tf.distributeVertically();
}

void TransferFunctionEditor::on_levelButton_clicked()
{
	tf.makeLevel(ui->doubleSpinBox->value());
}

void TransferFunctionEditor::on_diagonalButton_clicked()
{
	tf.makeDiagonal();
}

void TransferFunctionEditor::on_peaksButton_clicked()
{

}

void TransferFunctionEditor::on_rampButton_clicked()
{
	tf.makeRamp(ui->doubleSpinBox->value());
}

void TransferFunctionEditor::on_entropyButton_clicked()
{
#ifndef NOT_USED_BY_VOLUME_RENDERER
	if (tf.transfer_function)
	{
		tf.updateTransferFunctionFromView();

		tf.transfer_function->intensityOptimizerV2->numIterations = ui->spinBox->value();
		tf.transfer_function->intensityOptimizerV2->BalanceEdges();
		tf.transfer_function->LoadLookup(tf.transfer_function->currentColorTable);

		tf.updateViewFromTransferFunction();
	}
#endif // NOT_USED_BY_VOLUME_RENDERER
}

void TransferFunctionEditor::on_visibilityButton_clicked()
{
#ifndef NOT_USED_BY_VOLUME_RENDERER
	if (tf.transfer_function)
	{
		tf.updateTransferFunctionFromView();

		tf.transfer_function->intensityOptimizerV2->numIterations = ui->spinBox->value();
		tf.transfer_function->intensityOptimizerV2->BalanceVisibility();
		tf.transfer_function->LoadLookup(tf.transfer_function->currentColorTable);

		tf.updateViewFromTransferFunction();
	}
#endif // NOT_USED_BY_VOLUME_RENDERER
}

void TransferFunctionEditor::on_checkBox_clicked()
{
	ui->checkBox_2->setChecked(false);
	tf.is_ma_optimizer_enable = ui->checkBox->isChecked();
	tf.is_luo_optimizer_enable = false;
	std::cout << "Ma's optimizer " << (tf.isMaOptimizerEnable() ? "enabled" : "disabled") << std::endl;
}

void TransferFunctionEditor::on_checkBox_2_clicked()
{
	ui->checkBox->setChecked(false);
	tf.is_luo_optimizer_enable = ui->checkBox_2->isChecked();
	tf.is_ma_optimizer_enable = false;
	std::cout << "Luo's optimizer " << (tf.isLuoOptimizerEnable() ? "enabled" : "disabled") << std::endl;
}
