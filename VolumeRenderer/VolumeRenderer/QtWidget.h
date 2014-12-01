#ifndef QTWIDGET_H
#define QTWIDGET_H

#include <QtWidgets/QMainWindow>
#include "ui_volumerenderer.h"
#include <math.h>
#include "VolumeRenderer.h"
#include "TransferFunction.h"


class QtWidget : public QMainWindow
{
	Q_OBJECT



public:
	QtWidget(QWidget *parent = 0);
	~QtWidget();

	void Init(VolumeRenderer &VolumeRenderer);

	void InitOpacityTab();
	void InitRegionsTab();
	void InitRaycastTab();
	void InitShadersTab();
	void InitTimingTab();
	void InitContoursTab();
	void InitTransferFuncTab();

public slots:
	void ToggleGrabRegion(bool x);
	void CloseProgram();

#pragma region RegionsTab
	void AdjustCutOff(int x);
	void AdjustMinimum(int x);
	void AdjustMaximum(int x);
#pragma endregion RegionsTab

#pragma region TimingTab
	void AdjustTiming(int x);
#pragma endregion TimingTab

#pragma region ShadersTab
	void ChangeShader(QString qStr);
#pragma endregion ShadersTab
	
#pragma region RaycastTab
	void ChangeMaxRaySteps(QString qStr);
	void ChangeRayStepSize(QString qStr);
	void ChangeGradientStepSize(QString qStr);
#pragma endregion RaycastTab

#pragma region OpacityTab
	void ChangeOpacityDiv1Min(QString qStr);
	void ChangeOpacityDiv1Max(QString qStr);
	void ChangeOpacityDiv2Min(QString qStr);
	void ChangeOpacityDiv2Max(QString qStr);
	void ChangeOpacityDiv3Min(QString qStr);
	void ChangeOpacityDiv3Max(QString qStr);
	void ChangeOpacityDiv4Min(QString qStr);
	void ChangeOpacityDiv4Max(QString qStr);

	void AdjustOpacityDiv1(int x);
	void AdjustOpacityDiv2(int x);
	void AdjustOpacityDiv3(int x);
	void AdjustOpacityDiv4(int x);
#pragma endregion OpacityTab

#pragma region ContoursTab
	void AdjustContourThreshold(int x);
	void AdjustSuggestiveThreshold(int x);
	void AdjustNumPixelsLower(int x);
	void AdjustKernelRadius(int x);
	void ToggleShowDiffuse(bool x);
#pragma endregion ContoursTab

#pragma region TransferFuncTab
	void AdjustTFIntensity(int x);
	void ClampTFSliders(int x, int currentSlider, int direction);
	void AdjustIntensityFocus(int x);
	void ToggleIntensityOptimize(bool x);
#pragma endregion TransferFuncTab

private:
	VolumeRenderer *VolumeRendererPtr;
	VolumeDataset *volume;
	ShaderManager *shaderManager;
	Raycaster *raycaster;
	ContourDrawer *contourDrawer;
	TransferFunction *transferFunction;

	std::vector<QSlider*> tfIntensitySliders;
	std::vector<QLabel*> tfIntensityLabels;

	Ui::VolumeRendererClass ui;
};


#endif
