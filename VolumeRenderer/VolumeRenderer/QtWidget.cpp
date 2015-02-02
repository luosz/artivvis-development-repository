#include "QtWidget.h"

QtWidget::QtWidget(QWidget *parent)		:	  QMainWindow(parent)
{
	ui.setupUi(this);
}

QtWidget::~QtWidget()
{ }

void QtWidget::Init(VolumeRenderer &volumeRenderer)
{
	VolumeRendererPtr = &volumeRenderer;
	volume = &volumeRenderer.volume;
	shaderManager = &volumeRenderer.shaderManager;
	raycaster = volumeRenderer.renderer->raycaster;
	transferFunction = &volumeRenderer.renderer->transferFunction;
	contourDrawer = new GPUContours();									// Careful of this

	InitTransferFuncTab();
	InitRegionsTab();
	InitShadersTab();
	InitContoursTab();
	InitOpacityTab();
	InitRaycastTab();
	InitTimingTab();
}

void QtWidget::ToggleGrabRegion(bool x)
{
	VolumeRendererPtr->grabRegion = x;
}

void QtWidget::CloseProgram()
{
	cudaDeviceReset();
	exit(0);
}

#pragma region TransferFuncTab

void QtWidget::InitTransferFuncTab()
{
	tfIntensitySliders.push_back(ui.tfIntensity1);
	tfIntensitySliders.push_back(ui.tfIntensity2);
	tfIntensitySliders.push_back(ui.tfIntensity3);
	tfIntensitySliders.push_back(ui.tfIntensity4);
	tfIntensitySliders.push_back(ui.tfIntensity5);
	tfIntensitySliders.push_back(ui.tfIntensity6);
	tfIntensitySliders.push_back(ui.tfIntensity7);
	tfIntensitySliders.push_back(ui.tfIntensity8);
	tfIntensitySliders.push_back(ui.tfIntensity9);
	tfIntensitySliders.push_back(ui.tfIntensity10);
	tfIntensitySliders.push_back(ui.tfIntensity11);
	tfIntensitySliders.push_back(ui.tfIntensity12);
	tfIntensitySliders.push_back(ui.tfIntensity13);
//	tfIntensitySliders.push_back(ui.tfIntensity14);
//	tfIntensitySliders.push_back(ui.tfIntensity15);

	tfIntensityLabels.push_back(ui.tfIntLabel1);
	tfIntensityLabels.push_back(ui.tfIntLabel2);
	tfIntensityLabels.push_back(ui.tfIntLabel3);
	tfIntensityLabels.push_back(ui.tfIntLabel4);
	tfIntensityLabels.push_back(ui.tfIntLabel5);
	tfIntensityLabels.push_back(ui.tfIntLabel6);
	tfIntensityLabels.push_back(ui.tfIntLabel7);
	tfIntensityLabels.push_back(ui.tfIntLabel8);
	tfIntensityLabels.push_back(ui.tfIntLabel9);
	tfIntensityLabels.push_back(ui.tfIntLabel10);
	tfIntensityLabels.push_back(ui.tfIntLabel11);
	tfIntensityLabels.push_back(ui.tfIntLabel12);
	tfIntensityLabels.push_back(ui.tfIntLabel13);
//	tfIntensityLabels.push_back(ui.tfIntLabel14);
//	tfIntensityLabels.push_back(ui.tfIntLabel15);

	for (int i=0; i<tfIntensitySliders.size(); i++)
	{
		if (i<transferFunction->numIntensities)
		{
			tfIntensitySliders[i]->setValue(transferFunction->intensities[i] * 100);
			tfIntensityLabels[i]->setText(QString::number(transferFunction->intensities[i], 'f', 3));
		}
		else
			tfIntensityLabels[i]->setText("Null");
	}

	ui.tfIntensity15->setValue(transferFunction->targetIntensity * 100);
	ui.tfIntLabel15->setText(QString::number(transferFunction->targetIntensity, 'f', 3));
}

void QtWidget::ClampTFSliders(int x, int currentSlider, int direction)
{
	if (currentSlider < 0 || currentSlider >= tfIntensitySliders.size() || currentSlider >= transferFunction->numIntensities)
		return;

	if (x < tfIntensitySliders[currentSlider]->value() && direction == -1)
	{
		transferFunction->intensities[currentSlider] = x / 100.0f;
		tfIntensityLabels[currentSlider]->setText(QString::number(transferFunction->intensities[currentSlider], 'f', 3));
		tfIntensitySliders[currentSlider]->setValue(x);
		ClampTFSliders(x, currentSlider-1, -1);
	}
	else if (x > tfIntensitySliders[currentSlider]->value() && direction == 1)
	{
		transferFunction->intensities[currentSlider] = x / 100.0f;
		tfIntensityLabels[currentSlider]->setText(QString::number(transferFunction->intensities[currentSlider], 'f', 3));
		tfIntensitySliders[currentSlider]->setValue(x);
		ClampTFSliders(x, currentSlider+1, 1);
	}
}

void QtWidget::AdjustTFIntensity(int x)
{
	QObject *q = sender();

	for (int i=0; i<transferFunction->numIntensities && i<tfIntensitySliders.size(); i++)
	{
		if (q == tfIntensitySliders[i])
		{
			transferFunction->intensities[i] = x / 100.0f;
			tfIntensityLabels[i]->setText(QString::number(transferFunction->intensities[i], 'f', 3));

			ClampTFSliders(x, i-1, -1);
			ClampTFSliders(x, i+1, 1);
			break;
		}
	}
//	transferFunction->LoadLookup();
}

void QtWidget::AdjustIntensityFocus(int x)
{
	transferFunction->targetIntensity = x / 100.0f;
	ui.tfIntLabel15->setText(QString::number(transferFunction->targetIntensity, 'f', 3));

//	transferFunction->IntensityOptimize();
}

void QtWidget::ToggleIntensityOptimize(bool x)
{
	transferFunction->optimizeIntensity = x;
}

#pragma endregion TransferFuncTab


#pragma region RegionsTab

void QtWidget::InitRegionsTab()
{
	ui.cutOffSlider->setValue(raycaster->cutOff * 100);
	ui.cutOffLabel->setText(QString::number(raycaster->cutOff, 'f', 2));

	ui.minSlider->setValue(raycaster->minRange * 100);
	ui.minLabel->setText(QString::number(raycaster->minRange, 'f', 2));

	ui.maxSlider->setValue(raycaster->maxRange * 100);
	ui.maxLabel->setText(QString::number(raycaster->maxRange, 'f', 2));
}

void QtWidget::AdjustCutOff(int x)
{
	raycaster->cutOff = x / 100.0f;
	ui.cutOffLabel->setText(QString::number(raycaster->cutOff, 'f', 2));
	ui.cutOffSlider->setValue(x);

	if (raycaster->maxRange < raycaster->cutOff)
		AdjustMaximum(x);

	if (raycaster->minRange > raycaster->cutOff)
		AdjustMinimum(x);

}

void QtWidget::AdjustMinimum(int x)
{
	raycaster->minRange = x / 100.0f;
	ui.minLabel->setText(QString::number(raycaster->minRange, 'f', 2));
	ui.minSlider->setValue(x);

	if (raycaster->maxRange < raycaster->minRange)
		AdjustMaximum(x);

	if (raycaster->cutOff < raycaster->minRange)
		AdjustCutOff(x);
}

void QtWidget::AdjustMaximum(int x)
{
	raycaster->maxRange = x / 100.0f;
	ui.maxLabel->setText(QString::number(raycaster->maxRange, 'f', 2));
	ui.maxSlider->setValue(x);

	if (raycaster->cutOff > raycaster->maxRange)
		AdjustCutOff(x);

	if (raycaster->minRange > raycaster->maxRange)
		AdjustMinimum(x);
}

#pragma endregion RegionsTab


#pragma region ShadersTab

void QtWidget::InitShadersTab()
{
}

void QtWidget::ChangeShader(QString qStr)
{
	if (qStr == "Raycast Shader")
		shaderManager->currentShader = RaycastShader;
	else if (qStr == "Lighting Shader")
		shaderManager->currentShader = LightingShader;
	else if (qStr == "Opacity Shader")
		shaderManager->currentShader = OpacityShader;
	else if (qStr == "Depth Shader")
		shaderManager->currentShader = DepthShader;
	else if (qStr == "Normals Shader")
		shaderManager->currentShader = NormalsShader;
	else if (qStr == "X-Toon Shader")
		shaderManager->currentShader = XToonShader;
	else if (qStr == "Shadow Shader")
		shaderManager->currentShader = ShadowShader;
	else if (qStr == "Smoke Shader")
		shaderManager->currentShader = SmokeShader;
	else if (qStr == "Transfer Func Shader")
		shaderManager->currentShader = TFShader;
	else if (qStr == "TransFuncXtoon")
		shaderManager->currentShader = TransFuncXtoonShader;
}

#pragma endregion ShadersTab


#pragma region TimingTab

void QtWidget::InitTimingTab()
{
	ui.timingSlider->setValue(volume->timePerFrame * 100);
	ui.timingLabel->setText(QString::number(volume->timePerFrame, 'f', 2));	
}

void QtWidget::AdjustTiming(int x)
{
	volume->timePerFrame = x / 100.0f;
	ui.timingLabel->setText(QString::number(volume->timePerFrame, 'f', 2));
}

#pragma endregion TimingTab


#pragma region RaycastTab

void QtWidget::InitRaycastTab()
{
	ui.maxRayStepsBox->setText(QString::number(raycaster->maxRaySteps, 'f', 0));
	ui.rayStepSizeBox->setText(QString::number(raycaster->rayStepSize, 'f', 3));
	ui.gradientStepSizeBox->setText(QString::number(raycaster->gradientStepSize, 'f', 3));
}

void QtWidget::ChangeMaxRaySteps(QString qStr)
{
	raycaster->maxRaySteps = qStr.toInt();
}

void QtWidget::ChangeRayStepSize(QString qStr)
{
	raycaster->rayStepSize = qStr.toFloat();
}

void QtWidget::ChangeGradientStepSize(QString qStr)
{
	raycaster->gradientStepSize = qStr.toFloat();
}

#pragma endregion RaycastTab


#pragma region OpacityTab

void QtWidget::InitOpacityTab()
{
	ui.opacityDiv1Slider->setValue(raycaster->opacities[0] * 100);
	ui.opacityDiv1Label->setText(QString::number(raycaster->opacities[0], 'f', 2));

	ui.opacityDiv2Slider->setValue(raycaster->opacities[1] * 100);
	ui.opacityDiv2Label->setText(QString::number(raycaster->opacities[1], 'f', 2));

	ui.opacityDiv3Slider->setValue(raycaster->opacities[2] * 100);
	ui.opacityDiv3Label->setText(QString::number(raycaster->opacities[2], 'f', 2));

	ui.opacityDiv4Slider->setValue(raycaster->opacities[3] * 100);
	ui.opacityDiv4Label->setText(QString::number(raycaster->opacities[3], 'f', 2));
}

void QtWidget::ChangeOpacityDiv1Min(QString qStr)
{
	raycaster->opacityDivisions[0].x = qStr.toFloat();
}

void QtWidget::ChangeOpacityDiv1Max(QString qStr)
{
	raycaster->opacityDivisions[0].y = qStr.toFloat();
}

void QtWidget::ChangeOpacityDiv2Min(QString qStr)
{
	raycaster->opacityDivisions[1].x = qStr.toFloat();
}

void QtWidget::ChangeOpacityDiv2Max(QString qStr)
{
	raycaster->opacityDivisions[1].y = qStr.toFloat();
}

void QtWidget::ChangeOpacityDiv3Min(QString qStr)
{
	raycaster->opacityDivisions[2].x = qStr.toFloat();
}

void QtWidget::ChangeOpacityDiv3Max(QString qStr)
{
	raycaster->opacityDivisions[2].y = qStr.toFloat();
}

void QtWidget::ChangeOpacityDiv4Min(QString qStr)
{
	raycaster->opacityDivisions[3].x = qStr.toFloat();
}

void QtWidget::ChangeOpacityDiv4Max(QString qStr)
{
	raycaster->opacityDivisions[3].y = qStr.toFloat();
}

void QtWidget::AdjustOpacityDiv1(int x)
{
	raycaster->opacities[0] = x / 100.0f;
	ui.opacityDiv1Label->setText(QString::number(raycaster->opacities[0], 'f', 4));
	ui.opacityDiv1Slider->setValue(x);
}

void QtWidget::AdjustOpacityDiv2(int x)
{
	raycaster->opacities[1] = x / 100.0f;
	ui.opacityDiv2Label->setText(QString::number(raycaster->opacities[1], 'f', 4));
}

void QtWidget::AdjustOpacityDiv3(int x)
{
	raycaster->opacities[2] = x / 100.0f;
	ui.opacityDiv3Label->setText(QString::number(raycaster->opacities[2], 'f', 4));
}

void QtWidget::AdjustOpacityDiv4(int x)
{
	raycaster->opacities[3] = x / 100.0f;
	ui.opacityDiv4Label->setText(QString::number(raycaster->opacities[3], 'f', 4));
}

#pragma endregion OpacityTab


#pragma region ContoursTab

void QtWidget::InitContoursTab()
{
	ui.suggestiveThresholdSlider->setValue(contourDrawer->suggestiveContourThreshold * 200);
	ui.suggestiveThresholdLabel->setText(QString::number(contourDrawer->suggestiveContourThreshold, 'f', 2));

	ui.numPixelsLowerSlider->setValue(contourDrawer->numPixelsLower);
	ui.numPixelsLowerLabel->setText(QString::number(contourDrawer->numPixelsLower, 10));

	ui.kernelRadiusSlider->setValue(contourDrawer->kernelRadius);
	ui.kernelRadiusLabel->setText(QString::number(contourDrawer->kernelRadius, 10));
}

void QtWidget::AdjustContourThreshold(int x)
{
//	raycaster->contourThreshold = x / 100.0f;
//	ui.contourThresholdLabel->setText(QString::number(raycaster->contourThreshold, 'f', 2));
}

void QtWidget::AdjustSuggestiveThreshold(int x)
{
	contourDrawer->suggestiveContourThreshold = x / 200.0f;
	ui.suggestiveThresholdLabel->setText(QString::number(contourDrawer->suggestiveContourThreshold, 'f', 2));
}

void QtWidget::AdjustNumPixelsLower(int x)
{
	contourDrawer->numPixelsLower = x;
	ui.numPixelsLowerLabel->setText(QString::number(contourDrawer->numPixelsLower, 10));
}

void QtWidget::AdjustKernelRadius(int x)
{
	contourDrawer->kernelRadius = x;
	ui.kernelRadiusLabel->setText(QString::number(contourDrawer->kernelRadius, 10));
}

void QtWidget::ToggleShowDiffuse(bool x)
{

}

#pragma endregion ContoursTab


