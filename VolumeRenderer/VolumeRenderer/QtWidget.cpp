#include "QtWidget.h"

QtWidget::QtWidget(QWidget *parent)		:	  QMainWindow(parent)
{
	ui.setupUi(this);
}

QtWidget::~QtWidget()
{
}

void QtWidget::Init(VolumeRenderer &volumeRenderer)
{
	volume = &volumeRenderer.volume;
	shaderManager = &volumeRenderer.shaderManager;
	raycaster = volumeRenderer.renderer->raycaster;
	contourDrawer = volumeRenderer.renderer->contourDrawer;

	// Tab 1
	oldCutOffSliderPos = raycaster->cutOff * 100;
	ui.cutOffSlider->setValue(oldCutOffSliderPos);
	ui.cutOffLabel->setText(QString::number(raycaster->cutOff, 'f', 2));

	oldMinSliderPos = raycaster->minRange * 100;
	ui.minSlider->setValue(oldMinSliderPos);
	ui.minLabel->setText(QString::number(raycaster->minRange, 'f', 2));

	oldMaxSliderPos = raycaster->maxRange * 100;
	ui.maxSlider->setValue(oldMaxSliderPos);
	ui.maxLabel->setText(QString::number(raycaster->maxRange, 'f', 2));

	// Tab 2
	ui.maxRayStepsBox->setText(QString::number(raycaster->maxRaySteps, 'f', 0));
	ui.rayStepSizeBox->setText(QString::number(raycaster->rayStepSize, 'f', 3));
	ui.gradientStepSizeBox->setText(QString::number(raycaster->gradientStepSize, 'f', 3));
	

	// Tab 4
	oldTimingSliderPos = volume->timePerFrame * 100;
	ui.timingSlider->setValue(oldTimingSliderPos);
	ui.timingLabel->setText(QString::number(volume->timePerFrame, 'f', 2));	

	// Tab 5
	oldOpacityDiv1SliderPos = raycaster->opacities[0] * 100;
	ui.opacityDiv1Slider->setValue(oldOpacityDiv1SliderPos);
	ui.opacityDiv1Label->setText(QString::number(raycaster->opacities[0], 'f', 2));

	oldOpacityDiv2SliderPos = raycaster->opacities[1] * 100;
	ui.opacityDiv2Slider->setValue(oldOpacityDiv2SliderPos);
	ui.opacityDiv2Label->setText(QString::number(raycaster->opacities[1], 'f', 2));

	oldOpacityDiv3SliderPos = raycaster->opacities[2] * 100;
	ui.opacityDiv3Slider->setValue(oldOpacityDiv3SliderPos);
	ui.opacityDiv3Label->setText(QString::number(raycaster->opacities[2], 'f', 2));

	oldOpacityDiv4SliderPos = raycaster->opacities[3] * 100;
	ui.opacityDiv4Slider->setValue(oldOpacityDiv4SliderPos);
	ui.opacityDiv4Label->setText(QString::number(raycaster->opacities[3], 'f', 2));


	// Contour Tab
	ui.suggestiveThresholdSlider->setValue(contourDrawer->suggestiveContourThreshold * 200);
	ui.suggestiveThresholdLabel->setText(QString::number(contourDrawer->suggestiveContourThreshold, 'f', 2));

	ui.numPixelsLowerSlider->setValue(contourDrawer->numPixelsLower);
	ui.numPixelsLowerLabel->setText(QString::number(contourDrawer->numPixelsLower, 10));

	ui.kernelRadiusSlider->setValue(contourDrawer->kernelRadius);
	ui.kernelRadiusLabel->setText(QString::number(contourDrawer->kernelRadius, 10));
}


void QtWidget::CloseProgram()
{
	exit(0);
}

void QtWidget::AdjustCutOff(int x)
{
	raycaster->cutOff += (x - oldCutOffSliderPos) / 100.0f;
	oldCutOffSliderPos = x;
	ui.cutOffLabel->setText(QString::number(raycaster->cutOff, 'f', 2));
	ui.cutOffSlider->setValue(x);

	if (oldMaxSliderPos < oldCutOffSliderPos)
		AdjustMaximum(x);

	if (oldMinSliderPos > oldCutOffSliderPos)
		AdjustMinimum(x);

}

void QtWidget::AdjustMinimum(int x)
{
	raycaster->minRange += (x - oldMinSliderPos) / 100.0f;
	oldMinSliderPos = x;
	ui.minLabel->setText(QString::number(raycaster->minRange, 'f', 2));
	ui.minSlider->setValue(x);

	if (oldMaxSliderPos < oldMinSliderPos)
		AdjustMaximum(x);

	if (oldCutOffSliderPos < oldMinSliderPos)
		AdjustCutOff(x);
}

void QtWidget::AdjustMaximum(int x)
{
	raycaster->maxRange += (x - oldMaxSliderPos) / 100.0f;
	oldMaxSliderPos = x;
	ui.maxLabel->setText(QString::number(raycaster->maxRange, 'f', 2));
	ui.maxSlider->setValue(x);

	if (oldCutOffSliderPos > oldMaxSliderPos)
		AdjustCutOff(x);

	if (oldMinSliderPos > oldMaxSliderPos)
		AdjustMinimum(x);
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
}


void QtWidget::AdjustTiming(int x)
{
	volume->timePerFrame += (x - oldTimingSliderPos) / 100.0f;
	oldTimingSliderPos = x;
	ui.timingLabel->setText(QString::number(volume->timePerFrame, 'f', 2));
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