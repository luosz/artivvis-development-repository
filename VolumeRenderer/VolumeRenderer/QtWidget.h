#ifndef QTWIDGET_H
#define QTWIDGET_H

#include <QtWidgets/QMainWindow>
#include "ui_volumerenderer.h"
#include <math.h>
#include "VolumeRenderer.h"


class QtWidget : public QMainWindow
{
	Q_OBJECT



public:
	QtWidget(QWidget *parent = 0);
	~QtWidget();

	void Init(VolumeRenderer &VolumeRenderer);

public slots:
	void CloseProgram();
	void AdjustCutOff(int x);
	void AdjustMinimum(int x);
	void AdjustMaximum(int x);
	void AdjustTiming(int x);

	void ChangeShader(QString qStr);
	
	void ChangeMaxRaySteps(QString qStr);
	void ChangeRayStepSize(QString qStr);
	void ChangeGradientStepSize(QString qStr);

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

	void AdjustContourThreshold(int x);
	void AdjustSuggestiveThreshold(int x);
	void AdjustNumPixelsLower(int x);
	void AdjustKernelRadius(int x);
	void ToggleShowDiffuse(bool x);

private:
	VolumeDataset *volume;
	ShaderManager *shaderManager;
	Raycaster *raycaster;
	ContourDrawer *contourDrawer;

	int oldCutOffSliderPos;
	int oldMinSliderPos;
	int oldMaxSliderPos;
	int oldTimingSliderPos;

	int oldOpacityDiv1SliderPos;
	int oldOpacityDiv2SliderPos;
	int oldOpacityDiv3SliderPos;
	int oldOpacityDiv4SliderPos;

	Ui::VolumeRendererClass ui;
};


#endif
