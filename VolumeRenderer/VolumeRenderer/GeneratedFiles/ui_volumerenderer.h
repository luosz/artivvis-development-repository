/********************************************************************************
** Form generated from reading UI file 'volumerenderer.ui'
**
** Created by: Qt User Interface Compiler version 5.4.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VOLUMERENDERER_H
#define UI_VOLUMERENDERER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VolumeRendererClass
{
public:
    QWidget *centralWidget;
    QPushButton *pushButton;
    QTabWidget *tabWidget;
    QWidget *tab_5;
    QLineEdit *opacityDiv1Box1;
    QLabel *label_8;
    QLineEdit *opacityDiv1Box2;
    QLabel *label_9;
    QSlider *opacityDiv1Slider;
    QLabel *opacityDiv1Label;
    QLineEdit *opacityDiv2Box1;
    QLabel *opacityDiv2Label;
    QSlider *opacityDiv2Slider;
    QLineEdit *opacityDiv2Box2;
    QLineEdit *opacityDiv3Box1;
    QLabel *opacityDiv3Label;
    QSlider *opacityDiv3Slider;
    QLineEdit *opacityDiv3Box2;
    QLineEdit *opacityDiv4Box1;
    QLabel *opacityDiv4Label;
    QSlider *opacityDiv4Slider;
    QLineEdit *opacityDiv4Box2;
    QWidget *tab;
    QLabel *cutOffLabel;
    QLabel *maxLabel;
    QSlider *maxSlider;
    QLabel *label;
    QLabel *minLabel;
    QSlider *minSlider;
    QLabel *label_3;
    QSlider *cutOffSlider;
    QLabel *label_2;
    QWidget *tab_2;
    QLabel *label_5;
    QLabel *label_6;
    QLabel *label_7;
    QLineEdit *rayStepSizeBox;
    QLineEdit *maxRayStepsBox;
    QLineEdit *gradientStepSizeBox;
    QWidget *tab_3;
    QComboBox *shaderComboBox;
    QWidget *tab_4;
    QSlider *timingSlider;
    QLabel *label_4;
    QLabel *timingLabel;
    QWidget *tab_7;
    QSlider *contourThresholdSlider;
    QLabel *label_16;
    QLabel *contourThresholdLabel;
    QSlider *suggestiveThresholdSlider;
    QLabel *label_17;
    QLabel *suggestiveThresholdLabel;
    QSlider *numPixelsLowerSlider;
    QLabel *label_18;
    QLabel *numPixelsLowerLabel;
    QSlider *kernelRadiusSlider;
    QLabel *label_19;
    QLabel *kernelRadiusLabel;
    QCheckBox *showDiffuseCheckbox;
    QWidget *tab_6;
    QSlider *tfIntensity1;
    QSlider *tfIntensity2;
    QSlider *tfIntensity3;
    QSlider *tfIntensity4;
    QSlider *tfIntensity6;
    QSlider *tfIntensity9;
    QSlider *tfIntensity7;
    QSlider *tfIntensity8;
    QSlider *tfIntensity5;
    QSlider *tfIntensity10;
    QLabel *tfIntLabel1;
    QLabel *tfIntLabel2;
    QLabel *tfIntLabel3;
    QLabel *tfIntLabel4;
    QLabel *tfIntLabel5;
    QLabel *tfIntLabel8;
    QLabel *tfIntLabel6;
    QLabel *tfIntLabel7;
    QLabel *tfIntLabel9;
    QLabel *tfIntLabel10;
    QSlider *tfIntensity11;
    QSlider *tfIntensity12;
    QSlider *tfIntensity13;
    QSlider *tfIntensity15;
    QLabel *tfIntLabel12;
    QLabel *tfIntLabel15;
    QLabel *tfIntLabel13;
    QLabel *tfIntLabel11;
    QLabel *label_10;
    QCheckBox *checkBox;
    QCheckBox *GrabRegionCheckBox;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *VolumeRendererClass)
    {
        if (VolumeRendererClass->objectName().isEmpty())
            VolumeRendererClass->setObjectName(QStringLiteral("VolumeRendererClass"));
        VolumeRendererClass->setWindowModality(Qt::NonModal);
        VolumeRendererClass->resize(520, 341);
        centralWidget = new QWidget(VolumeRendererClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        pushButton = new QPushButton(centralWidget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(390, 260, 75, 23));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(30, 20, 461, 231));
        tab_5 = new QWidget();
        tab_5->setObjectName(QStringLiteral("tab_5"));
        opacityDiv1Box1 = new QLineEdit(tab_5);
        opacityDiv1Box1->setObjectName(QStringLiteral("opacityDiv1Box1"));
        opacityDiv1Box1->setGeometry(QRect(10, 40, 51, 20));
        label_8 = new QLabel(tab_5);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(50, 10, 46, 13));
        opacityDiv1Box2 = new QLineEdit(tab_5);
        opacityDiv1Box2->setObjectName(QStringLiteral("opacityDiv1Box2"));
        opacityDiv1Box2->setGeometry(QRect(90, 40, 51, 20));
        label_9 = new QLabel(tab_5);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(210, 10, 46, 13));
        opacityDiv1Slider = new QSlider(tab_5);
        opacityDiv1Slider->setObjectName(QStringLiteral("opacityDiv1Slider"));
        opacityDiv1Slider->setGeometry(QRect(160, 40, 141, 19));
        opacityDiv1Slider->setMaximum(100);
        opacityDiv1Slider->setOrientation(Qt::Horizontal);
        opacityDiv1Slider->setTickPosition(QSlider::TicksAbove);
        opacityDiv1Label = new QLabel(tab_5);
        opacityDiv1Label->setObjectName(QStringLiteral("opacityDiv1Label"));
        opacityDiv1Label->setGeometry(QRect(300, 40, 46, 13));
        opacityDiv2Box1 = new QLineEdit(tab_5);
        opacityDiv2Box1->setObjectName(QStringLiteral("opacityDiv2Box1"));
        opacityDiv2Box1->setGeometry(QRect(10, 80, 51, 20));
        opacityDiv2Label = new QLabel(tab_5);
        opacityDiv2Label->setObjectName(QStringLiteral("opacityDiv2Label"));
        opacityDiv2Label->setGeometry(QRect(300, 80, 46, 13));
        opacityDiv2Slider = new QSlider(tab_5);
        opacityDiv2Slider->setObjectName(QStringLiteral("opacityDiv2Slider"));
        opacityDiv2Slider->setGeometry(QRect(160, 80, 141, 19));
        opacityDiv2Slider->setMaximum(100);
        opacityDiv2Slider->setOrientation(Qt::Horizontal);
        opacityDiv2Slider->setTickPosition(QSlider::TicksAbove);
        opacityDiv2Box2 = new QLineEdit(tab_5);
        opacityDiv2Box2->setObjectName(QStringLiteral("opacityDiv2Box2"));
        opacityDiv2Box2->setGeometry(QRect(90, 80, 51, 20));
        opacityDiv3Box1 = new QLineEdit(tab_5);
        opacityDiv3Box1->setObjectName(QStringLiteral("opacityDiv3Box1"));
        opacityDiv3Box1->setGeometry(QRect(10, 120, 51, 20));
        opacityDiv3Label = new QLabel(tab_5);
        opacityDiv3Label->setObjectName(QStringLiteral("opacityDiv3Label"));
        opacityDiv3Label->setGeometry(QRect(300, 120, 46, 13));
        opacityDiv3Slider = new QSlider(tab_5);
        opacityDiv3Slider->setObjectName(QStringLiteral("opacityDiv3Slider"));
        opacityDiv3Slider->setGeometry(QRect(160, 120, 141, 19));
        opacityDiv3Slider->setMaximum(100);
        opacityDiv3Slider->setOrientation(Qt::Horizontal);
        opacityDiv3Slider->setTickPosition(QSlider::TicksAbove);
        opacityDiv3Box2 = new QLineEdit(tab_5);
        opacityDiv3Box2->setObjectName(QStringLiteral("opacityDiv3Box2"));
        opacityDiv3Box2->setGeometry(QRect(90, 120, 51, 20));
        opacityDiv4Box1 = new QLineEdit(tab_5);
        opacityDiv4Box1->setObjectName(QStringLiteral("opacityDiv4Box1"));
        opacityDiv4Box1->setGeometry(QRect(10, 160, 51, 20));
        opacityDiv4Label = new QLabel(tab_5);
        opacityDiv4Label->setObjectName(QStringLiteral("opacityDiv4Label"));
        opacityDiv4Label->setGeometry(QRect(300, 160, 46, 13));
        opacityDiv4Slider = new QSlider(tab_5);
        opacityDiv4Slider->setObjectName(QStringLiteral("opacityDiv4Slider"));
        opacityDiv4Slider->setGeometry(QRect(160, 160, 141, 19));
        opacityDiv4Slider->setMaximum(100);
        opacityDiv4Slider->setOrientation(Qt::Horizontal);
        opacityDiv4Slider->setTickPosition(QSlider::TicksAbove);
        opacityDiv4Box2 = new QLineEdit(tab_5);
        opacityDiv4Box2->setObjectName(QStringLiteral("opacityDiv4Box2"));
        opacityDiv4Box2->setGeometry(QRect(90, 160, 51, 20));
        tabWidget->addTab(tab_5, QString());
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        cutOffLabel = new QLabel(tab);
        cutOffLabel->setObjectName(QStringLiteral("cutOffLabel"));
        cutOffLabel->setGeometry(QRect(270, 90, 46, 16));
        maxLabel = new QLabel(tab);
        maxLabel->setObjectName(QStringLiteral("maxLabel"));
        maxLabel->setGeometry(QRect(270, 50, 46, 13));
        maxSlider = new QSlider(tab);
        maxSlider->setObjectName(QStringLiteral("maxSlider"));
        maxSlider->setGeometry(QRect(100, 50, 161, 24));
        maxSlider->setMaximum(100);
        maxSlider->setSliderPosition(50);
        maxSlider->setOrientation(Qt::Horizontal);
        maxSlider->setTickPosition(QSlider::TicksAbove);
        label = new QLabel(tab);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(50, 90, 41, 16));
        minLabel = new QLabel(tab);
        minLabel->setObjectName(QStringLiteral("minLabel"));
        minLabel->setGeometry(QRect(270, 130, 46, 13));
        minSlider = new QSlider(tab);
        minSlider->setObjectName(QStringLiteral("minSlider"));
        minSlider->setGeometry(QRect(100, 130, 161, 24));
        minSlider->setMaximum(100);
        minSlider->setSliderPosition(50);
        minSlider->setOrientation(Qt::Horizontal);
        minSlider->setTickPosition(QSlider::TicksAbove);
        label_3 = new QLabel(tab);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(50, 50, 46, 13));
        cutOffSlider = new QSlider(tab);
        cutOffSlider->setObjectName(QStringLiteral("cutOffSlider"));
        cutOffSlider->setGeometry(QRect(100, 90, 161, 24));
        cutOffSlider->setMaximum(100);
        cutOffSlider->setSliderPosition(50);
        cutOffSlider->setOrientation(Qt::Horizontal);
        cutOffSlider->setTickPosition(QSlider::TicksAbove);
        label_2 = new QLabel(tab);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(50, 130, 46, 13));
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        label_5 = new QLabel(tab_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(10, 120, 91, 20));
        label_6 = new QLabel(tab_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(30, 40, 71, 20));
        label_7 = new QLabel(tab_2);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(30, 80, 71, 20));
        rayStepSizeBox = new QLineEdit(tab_2);
        rayStepSizeBox->setObjectName(QStringLiteral("rayStepSizeBox"));
        rayStepSizeBox->setGeometry(QRect(110, 40, 113, 20));
        maxRayStepsBox = new QLineEdit(tab_2);
        maxRayStepsBox->setObjectName(QStringLiteral("maxRayStepsBox"));
        maxRayStepsBox->setGeometry(QRect(110, 80, 113, 20));
        gradientStepSizeBox = new QLineEdit(tab_2);
        gradientStepSizeBox->setObjectName(QStringLiteral("gradientStepSizeBox"));
        gradientStepSizeBox->setGeometry(QRect(110, 120, 113, 20));
        tabWidget->addTab(tab_2, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        shaderComboBox = new QComboBox(tab_3);
        shaderComboBox->setObjectName(QStringLiteral("shaderComboBox"));
        shaderComboBox->setGeometry(QRect(80, 30, 201, 21));
        shaderComboBox->setEditable(false);
        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QStringLiteral("tab_4"));
        timingSlider = new QSlider(tab_4);
        timingSlider->setObjectName(QStringLiteral("timingSlider"));
        timingSlider->setGeometry(QRect(100, 60, 160, 19));
        timingSlider->setOrientation(Qt::Horizontal);
        label_4 = new QLabel(tab_4);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(150, 30, 81, 20));
        timingLabel = new QLabel(tab_4);
        timingLabel->setObjectName(QStringLiteral("timingLabel"));
        timingLabel->setGeometry(QRect(270, 60, 46, 13));
        tabWidget->addTab(tab_4, QString());
        tab_7 = new QWidget();
        tab_7->setObjectName(QStringLiteral("tab_7"));
        contourThresholdSlider = new QSlider(tab_7);
        contourThresholdSlider->setObjectName(QStringLiteral("contourThresholdSlider"));
        contourThresholdSlider->setGeometry(QRect(110, 20, 281, 20));
        contourThresholdSlider->setMaximum(100);
        contourThresholdSlider->setOrientation(Qt::Horizontal);
        contourThresholdSlider->setTickPosition(QSlider::TicksAbove);
        label_16 = new QLabel(tab_7);
        label_16->setObjectName(QStringLiteral("label_16"));
        label_16->setGeometry(QRect(10, 20, 91, 16));
        contourThresholdLabel = new QLabel(tab_7);
        contourThresholdLabel->setObjectName(QStringLiteral("contourThresholdLabel"));
        contourThresholdLabel->setGeometry(QRect(400, 20, 31, 16));
        suggestiveThresholdSlider = new QSlider(tab_7);
        suggestiveThresholdSlider->setObjectName(QStringLiteral("suggestiveThresholdSlider"));
        suggestiveThresholdSlider->setGeometry(QRect(110, 50, 281, 20));
        suggestiveThresholdSlider->setMaximum(100);
        suggestiveThresholdSlider->setOrientation(Qt::Horizontal);
        suggestiveThresholdSlider->setTickPosition(QSlider::TicksAbove);
        label_17 = new QLabel(tab_7);
        label_17->setObjectName(QStringLiteral("label_17"));
        label_17->setGeometry(QRect(10, 50, 91, 16));
        suggestiveThresholdLabel = new QLabel(tab_7);
        suggestiveThresholdLabel->setObjectName(QStringLiteral("suggestiveThresholdLabel"));
        suggestiveThresholdLabel->setGeometry(QRect(400, 50, 31, 16));
        numPixelsLowerSlider = new QSlider(tab_7);
        numPixelsLowerSlider->setObjectName(QStringLiteral("numPixelsLowerSlider"));
        numPixelsLowerSlider->setGeometry(QRect(110, 80, 281, 20));
        numPixelsLowerSlider->setMaximum(100);
        numPixelsLowerSlider->setOrientation(Qt::Horizontal);
        numPixelsLowerSlider->setTickPosition(QSlider::TicksAbove);
        label_18 = new QLabel(tab_7);
        label_18->setObjectName(QStringLiteral("label_18"));
        label_18->setGeometry(QRect(10, 80, 91, 16));
        numPixelsLowerLabel = new QLabel(tab_7);
        numPixelsLowerLabel->setObjectName(QStringLiteral("numPixelsLowerLabel"));
        numPixelsLowerLabel->setGeometry(QRect(400, 80, 31, 16));
        kernelRadiusSlider = new QSlider(tab_7);
        kernelRadiusSlider->setObjectName(QStringLiteral("kernelRadiusSlider"));
        kernelRadiusSlider->setGeometry(QRect(110, 110, 281, 20));
        kernelRadiusSlider->setMaximum(100);
        kernelRadiusSlider->setOrientation(Qt::Horizontal);
        kernelRadiusSlider->setTickPosition(QSlider::TicksAbove);
        label_19 = new QLabel(tab_7);
        label_19->setObjectName(QStringLiteral("label_19"));
        label_19->setGeometry(QRect(10, 110, 91, 16));
        kernelRadiusLabel = new QLabel(tab_7);
        kernelRadiusLabel->setObjectName(QStringLiteral("kernelRadiusLabel"));
        kernelRadiusLabel->setGeometry(QRect(400, 110, 31, 16));
        showDiffuseCheckbox = new QCheckBox(tab_7);
        showDiffuseCheckbox->setObjectName(QStringLiteral("showDiffuseCheckbox"));
        showDiffuseCheckbox->setGeometry(QRect(200, 150, 91, 17));
        tabWidget->addTab(tab_7, QString());
        tab_6 = new QWidget();
        tab_6->setObjectName(QStringLiteral("tab_6"));
        tfIntensity1 = new QSlider(tab_6);
        tfIntensity1->setObjectName(QStringLiteral("tfIntensity1"));
        tfIntensity1->setGeometry(QRect(10, 10, 91, 19));
        tfIntensity1->setMaximum(100);
        tfIntensity1->setOrientation(Qt::Horizontal);
        tfIntensity1->setTickPosition(QSlider::TicksAbove);
        tfIntensity2 = new QSlider(tab_6);
        tfIntensity2->setObjectName(QStringLiteral("tfIntensity2"));
        tfIntensity2->setGeometry(QRect(10, 50, 91, 19));
        tfIntensity2->setMaximum(100);
        tfIntensity2->setOrientation(Qt::Horizontal);
        tfIntensity2->setTickPosition(QSlider::TicksAbove);
        tfIntensity3 = new QSlider(tab_6);
        tfIntensity3->setObjectName(QStringLiteral("tfIntensity3"));
        tfIntensity3->setGeometry(QRect(10, 90, 91, 19));
        tfIntensity3->setMaximum(100);
        tfIntensity3->setOrientation(Qt::Horizontal);
        tfIntensity3->setTickPosition(QSlider::TicksAbove);
        tfIntensity4 = new QSlider(tab_6);
        tfIntensity4->setObjectName(QStringLiteral("tfIntensity4"));
        tfIntensity4->setGeometry(QRect(10, 130, 91, 19));
        tfIntensity4->setMaximum(100);
        tfIntensity4->setOrientation(Qt::Horizontal);
        tfIntensity4->setTickPosition(QSlider::TicksAbove);
        tfIntensity6 = new QSlider(tab_6);
        tfIntensity6->setObjectName(QStringLiteral("tfIntensity6"));
        tfIntensity6->setGeometry(QRect(150, 10, 91, 19));
        tfIntensity6->setMaximum(100);
        tfIntensity6->setOrientation(Qt::Horizontal);
        tfIntensity6->setTickPosition(QSlider::TicksAbove);
        tfIntensity9 = new QSlider(tab_6);
        tfIntensity9->setObjectName(QStringLiteral("tfIntensity9"));
        tfIntensity9->setGeometry(QRect(150, 130, 91, 19));
        tfIntensity9->setMaximum(100);
        tfIntensity9->setOrientation(Qt::Horizontal);
        tfIntensity9->setTickPosition(QSlider::TicksAbove);
        tfIntensity7 = new QSlider(tab_6);
        tfIntensity7->setObjectName(QStringLiteral("tfIntensity7"));
        tfIntensity7->setGeometry(QRect(150, 50, 91, 19));
        tfIntensity7->setMaximum(100);
        tfIntensity7->setOrientation(Qt::Horizontal);
        tfIntensity7->setTickPosition(QSlider::TicksAbove);
        tfIntensity8 = new QSlider(tab_6);
        tfIntensity8->setObjectName(QStringLiteral("tfIntensity8"));
        tfIntensity8->setGeometry(QRect(150, 90, 91, 19));
        tfIntensity8->setMaximum(100);
        tfIntensity8->setOrientation(Qt::Horizontal);
        tfIntensity8->setTickPosition(QSlider::TicksAbove);
        tfIntensity5 = new QSlider(tab_6);
        tfIntensity5->setObjectName(QStringLiteral("tfIntensity5"));
        tfIntensity5->setGeometry(QRect(10, 170, 91, 19));
        tfIntensity5->setMaximum(100);
        tfIntensity5->setOrientation(Qt::Horizontal);
        tfIntensity5->setTickPosition(QSlider::TicksAbove);
        tfIntensity10 = new QSlider(tab_6);
        tfIntensity10->setObjectName(QStringLiteral("tfIntensity10"));
        tfIntensity10->setGeometry(QRect(150, 170, 91, 19));
        tfIntensity10->setMaximum(100);
        tfIntensity10->setOrientation(Qt::Horizontal);
        tfIntensity10->setTickPosition(QSlider::TicksAbove);
        tfIntLabel1 = new QLabel(tab_6);
        tfIntLabel1->setObjectName(QStringLiteral("tfIntLabel1"));
        tfIntLabel1->setGeometry(QRect(110, 10, 31, 16));
        tfIntLabel2 = new QLabel(tab_6);
        tfIntLabel2->setObjectName(QStringLiteral("tfIntLabel2"));
        tfIntLabel2->setGeometry(QRect(110, 50, 31, 16));
        tfIntLabel3 = new QLabel(tab_6);
        tfIntLabel3->setObjectName(QStringLiteral("tfIntLabel3"));
        tfIntLabel3->setGeometry(QRect(110, 90, 31, 16));
        tfIntLabel4 = new QLabel(tab_6);
        tfIntLabel4->setObjectName(QStringLiteral("tfIntLabel4"));
        tfIntLabel4->setGeometry(QRect(110, 130, 31, 16));
        tfIntLabel5 = new QLabel(tab_6);
        tfIntLabel5->setObjectName(QStringLiteral("tfIntLabel5"));
        tfIntLabel5->setGeometry(QRect(110, 170, 31, 16));
        tfIntLabel8 = new QLabel(tab_6);
        tfIntLabel8->setObjectName(QStringLiteral("tfIntLabel8"));
        tfIntLabel8->setGeometry(QRect(250, 90, 31, 16));
        tfIntLabel6 = new QLabel(tab_6);
        tfIntLabel6->setObjectName(QStringLiteral("tfIntLabel6"));
        tfIntLabel6->setGeometry(QRect(250, 10, 31, 16));
        tfIntLabel7 = new QLabel(tab_6);
        tfIntLabel7->setObjectName(QStringLiteral("tfIntLabel7"));
        tfIntLabel7->setGeometry(QRect(250, 50, 31, 16));
        tfIntLabel9 = new QLabel(tab_6);
        tfIntLabel9->setObjectName(QStringLiteral("tfIntLabel9"));
        tfIntLabel9->setGeometry(QRect(250, 130, 31, 16));
        tfIntLabel10 = new QLabel(tab_6);
        tfIntLabel10->setObjectName(QStringLiteral("tfIntLabel10"));
        tfIntLabel10->setGeometry(QRect(250, 170, 31, 16));
        tfIntensity11 = new QSlider(tab_6);
        tfIntensity11->setObjectName(QStringLiteral("tfIntensity11"));
        tfIntensity11->setGeometry(QRect(290, 10, 91, 19));
        tfIntensity11->setMaximum(100);
        tfIntensity11->setOrientation(Qt::Horizontal);
        tfIntensity11->setTickPosition(QSlider::TicksAbove);
        tfIntensity12 = new QSlider(tab_6);
        tfIntensity12->setObjectName(QStringLiteral("tfIntensity12"));
        tfIntensity12->setGeometry(QRect(290, 50, 91, 19));
        tfIntensity12->setMaximum(100);
        tfIntensity12->setOrientation(Qt::Horizontal);
        tfIntensity12->setTickPosition(QSlider::TicksAbove);
        tfIntensity13 = new QSlider(tab_6);
        tfIntensity13->setObjectName(QStringLiteral("tfIntensity13"));
        tfIntensity13->setGeometry(QRect(290, 90, 91, 19));
        tfIntensity13->setMaximum(100);
        tfIntensity13->setOrientation(Qt::Horizontal);
        tfIntensity13->setTickPosition(QSlider::TicksAbove);
        tfIntensity15 = new QSlider(tab_6);
        tfIntensity15->setObjectName(QStringLiteral("tfIntensity15"));
        tfIntensity15->setGeometry(QRect(310, 170, 101, 20));
        tfIntensity15->setMaximum(100);
        tfIntensity15->setOrientation(Qt::Horizontal);
        tfIntensity15->setTickPosition(QSlider::TicksAbove);
        tfIntLabel12 = new QLabel(tab_6);
        tfIntLabel12->setObjectName(QStringLiteral("tfIntLabel12"));
        tfIntLabel12->setGeometry(QRect(390, 50, 31, 16));
        tfIntLabel15 = new QLabel(tab_6);
        tfIntLabel15->setObjectName(QStringLiteral("tfIntLabel15"));
        tfIntLabel15->setGeometry(QRect(420, 170, 31, 16));
        tfIntLabel13 = new QLabel(tab_6);
        tfIntLabel13->setObjectName(QStringLiteral("tfIntLabel13"));
        tfIntLabel13->setGeometry(QRect(390, 90, 31, 16));
        tfIntLabel11 = new QLabel(tab_6);
        tfIntLabel11->setObjectName(QStringLiteral("tfIntLabel11"));
        tfIntLabel11->setGeometry(QRect(390, 10, 31, 16));
        label_10 = new QLabel(tab_6);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(320, 150, 81, 16));
        checkBox = new QCheckBox(tab_6);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        checkBox->setGeometry(QRect(310, 130, 121, 17));
        tabWidget->addTab(tab_6, QString());
        GrabRegionCheckBox = new QCheckBox(centralWidget);
        GrabRegionCheckBox->setObjectName(QStringLiteral("GrabRegionCheckBox"));
        GrabRegionCheckBox->setGeometry(QRect(80, 260, 81, 17));
        VolumeRendererClass->setCentralWidget(centralWidget);
        tabWidget->raise();
        pushButton->raise();
        GrabRegionCheckBox->raise();
        menuBar = new QMenuBar(VolumeRendererClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 520, 26));
        VolumeRendererClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(VolumeRendererClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        VolumeRendererClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(VolumeRendererClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        VolumeRendererClass->setStatusBar(statusBar);

        retranslateUi(VolumeRendererClass);
        QObject::connect(pushButton, SIGNAL(clicked()), VolumeRendererClass, SLOT(CloseProgram()));
        QObject::connect(maxSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustMaximum(int)));
        QObject::connect(cutOffSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustCutOff(int)));
        QObject::connect(minSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustMinimum(int)));
        QObject::connect(timingSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustTiming(int)));
        QObject::connect(rayStepSizeBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeRayStepSize(QString)));
        QObject::connect(maxRayStepsBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeMaxRaySteps(QString)));
        QObject::connect(gradientStepSizeBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeGradientStepSize(QString)));
        QObject::connect(shaderComboBox, SIGNAL(currentTextChanged(QString)), VolumeRendererClass, SLOT(ChangeShader(QString)));
        QObject::connect(opacityDiv1Box1, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv1Min(QString)));
        QObject::connect(opacityDiv2Box1, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv2Min(QString)));
        QObject::connect(opacityDiv3Box1, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv3Min(QString)));
        QObject::connect(opacityDiv1Box2, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv1Max(QString)));
        QObject::connect(opacityDiv2Box2, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv2Max(QString)));
        QObject::connect(opacityDiv3Box2, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv3Max(QString)));
        QObject::connect(opacityDiv1Slider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustOpacityDiv1(int)));
        QObject::connect(opacityDiv2Slider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustOpacityDiv2(int)));
        QObject::connect(opacityDiv3Slider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustOpacityDiv3(int)));
        QObject::connect(opacityDiv4Box1, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv4Min(QString)));
        QObject::connect(opacityDiv4Box2, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv4Max(QString)));
        QObject::connect(opacityDiv4Slider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustOpacityDiv4(int)));
        QObject::connect(contourThresholdSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustContourThreshold(int)));
        QObject::connect(suggestiveThresholdSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustSuggestiveThreshold(int)));
        QObject::connect(numPixelsLowerSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustNumPixelsLower(int)));
        QObject::connect(kernelRadiusSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustKernelRadius(int)));
        QObject::connect(showDiffuseCheckbox, SIGNAL(toggled(bool)), VolumeRendererClass, SLOT(ToggleShowDiffuse(bool)));
        QObject::connect(tfIntensity1, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity2, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity3, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity4, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity5, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity6, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity7, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity8, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity9, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity10, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity11, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity12, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity13, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustTFIntensity(int)));
        QObject::connect(tfIntensity15, SIGNAL(sliderMoved(int)), VolumeRendererClass, SLOT(AdjustIntensityFocus(int)));
        QObject::connect(checkBox, SIGNAL(toggled(bool)), VolumeRendererClass, SLOT(ToggleIntensityOptimize(bool)));
        QObject::connect(GrabRegionCheckBox, SIGNAL(toggled(bool)), VolumeRendererClass, SLOT(ToggleGrabRegion(bool)));

        tabWidget->setCurrentIndex(3);
        shaderComboBox->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(VolumeRendererClass);
    } // setupUi

    void retranslateUi(QMainWindow *VolumeRendererClass)
    {
        VolumeRendererClass->setWindowTitle(QApplication::translate("VolumeRendererClass", "VolumeRenderer", 0));
        pushButton->setText(QApplication::translate("VolumeRendererClass", "Quit", 0));
        label_8->setText(QApplication::translate("VolumeRendererClass", "Divisions", 0));
        label_9->setText(QApplication::translate("VolumeRendererClass", "Opacity", 0));
        opacityDiv1Label->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        opacityDiv2Label->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        opacityDiv3Label->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        opacityDiv4Label->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_5), QApplication::translate("VolumeRendererClass", "Opacity", 0));
        cutOffLabel->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        maxLabel->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        label->setText(QApplication::translate("VolumeRendererClass", "CutOff", 0));
        minLabel->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        label_3->setText(QApplication::translate("VolumeRendererClass", "Maximum", 0));
        label_2->setText(QApplication::translate("VolumeRendererClass", "Minimum", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("VolumeRendererClass", "Regions", 0));
        label_5->setText(QApplication::translate("VolumeRendererClass", "Gradient Step Size", 0));
        label_6->setText(QApplication::translate("VolumeRendererClass", "Ray Step Size", 0));
        label_7->setText(QApplication::translate("VolumeRendererClass", "Max Ray Steps", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("VolumeRendererClass", "Raycast", 0));
        shaderComboBox->clear();
        shaderComboBox->insertItems(0, QStringList()
         << QApplication::translate("VolumeRendererClass", "Transfer Func Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Smoke Shader", 0)
         << QApplication::translate("VolumeRendererClass", "X-Toon Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Raycast Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Lighting Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Opacity Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Depth Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Normals Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Shadow Shader", 0)
         << QApplication::translate("VolumeRendererClass", "TransFuncXtoon", 0)
        );
        shaderComboBox->setCurrentText(QApplication::translate("VolumeRendererClass", "Transfer Func Shader", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("VolumeRendererClass", "Shaders", 0));
        label_4->setText(QApplication::translate("VolumeRendererClass", "Time Per Frame", 0));
        timingLabel->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QApplication::translate("VolumeRendererClass", "Timing", 0));
        label_16->setText(QApplication::translate("VolumeRendererClass", "Contour Threshold", 0));
        contourThresholdLabel->setText(QApplication::translate("VolumeRendererClass", "0.00", 0));
        label_17->setText(QApplication::translate("VolumeRendererClass", "Suggestive Thresh", 0));
        suggestiveThresholdLabel->setText(QApplication::translate("VolumeRendererClass", "0.00", 0));
        label_18->setText(QApplication::translate("VolumeRendererClass", "Num Pixels Lower", 0));
        numPixelsLowerLabel->setText(QApplication::translate("VolumeRendererClass", "0.00", 0));
        label_19->setText(QApplication::translate("VolumeRendererClass", "Kernel Radius", 0));
        kernelRadiusLabel->setText(QApplication::translate("VolumeRendererClass", "0.00", 0));
        showDiffuseCheckbox->setText(QApplication::translate("VolumeRendererClass", "Show Diffuse", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_7), QApplication::translate("VolumeRendererClass", "Contours", 0));
        tfIntLabel1->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel2->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel3->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel4->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel5->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel8->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel6->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel7->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel9->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel10->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel12->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel15->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel13->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        tfIntLabel11->setText(QApplication::translate("VolumeRendererClass", "TextLabel", 0));
        label_10->setText(QApplication::translate("VolumeRendererClass", "Intensity Focus", 0));
        checkBox->setText(QApplication::translate("VolumeRendererClass", "Optimize Intensity", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_6), QApplication::translate("VolumeRendererClass", "TransferFunc", 0));
        GrabRegionCheckBox->setText(QApplication::translate("VolumeRendererClass", "GrabRegion", 0));
    } // retranslateUi

};

namespace Ui {
    class VolumeRendererClass: public Ui_VolumeRendererClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VOLUMERENDERER_H
