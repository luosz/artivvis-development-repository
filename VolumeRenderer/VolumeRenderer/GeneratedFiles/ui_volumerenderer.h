/********************************************************************************
** Form generated from reading UI file 'volumerenderer.ui'
**
** Created by: Qt User Interface Compiler version 5.3.2
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
#include <QtWidgets/QFrame>
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
    QLineEdit *div3MinBox;
    QLineEdit *div1MaxBox;
    QLineEdit *div1MinBox;
    QLineEdit *div4MinBox;
    QLineEdit *div4MaxBox;
    QLineEdit *div2MinBox;
    QLineEdit *div2MaxBox;
    QLineEdit *div3MaxBox;
    QSlider *div1MinSlider;
    QSlider *div2MinSlider;
    QSlider *div3MinSlider;
    QSlider *div4MinSlider;
    QSlider *div1MaxSlider;
    QSlider *div4MaxSlider;
    QSlider *div2MaxSlider;
    QSlider *div3MaxSlider;
    QFrame *line;
    QLabel *label_10;
    QLabel *label_11;
    QLabel *label_12;
    QLabel *label_13;
    QLabel *label_14;
    QLabel *label_15;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *VolumeRendererClass)
    {
        if (VolumeRendererClass->objectName().isEmpty())
            VolumeRendererClass->setObjectName(QStringLiteral("VolumeRendererClass"));
        VolumeRendererClass->setWindowModality(Qt::NonModal);
        VolumeRendererClass->resize(514, 341);
        centralWidget = new QWidget(VolumeRendererClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        pushButton = new QPushButton(centralWidget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(390, 260, 75, 23));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(30, 20, 451, 231));
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
        div3MinBox = new QLineEdit(tab_6);
        div3MinBox->setObjectName(QStringLiteral("div3MinBox"));
        div3MinBox->setGeometry(QRect(210, 130, 41, 20));
        div1MaxBox = new QLineEdit(tab_6);
        div1MaxBox->setObjectName(QStringLiteral("div1MaxBox"));
        div1MaxBox->setGeometry(QRect(400, 50, 41, 20));
        div1MinBox = new QLineEdit(tab_6);
        div1MinBox->setObjectName(QStringLiteral("div1MinBox"));
        div1MinBox->setGeometry(QRect(210, 50, 41, 20));
        div4MinBox = new QLineEdit(tab_6);
        div4MinBox->setObjectName(QStringLiteral("div4MinBox"));
        div4MinBox->setGeometry(QRect(210, 170, 41, 20));
        div4MaxBox = new QLineEdit(tab_6);
        div4MaxBox->setObjectName(QStringLiteral("div4MaxBox"));
        div4MaxBox->setGeometry(QRect(400, 170, 41, 20));
        div2MinBox = new QLineEdit(tab_6);
        div2MinBox->setObjectName(QStringLiteral("div2MinBox"));
        div2MinBox->setGeometry(QRect(210, 90, 41, 20));
        div2MaxBox = new QLineEdit(tab_6);
        div2MaxBox->setObjectName(QStringLiteral("div2MaxBox"));
        div2MaxBox->setGeometry(QRect(400, 90, 41, 20));
        div3MaxBox = new QLineEdit(tab_6);
        div3MaxBox->setObjectName(QStringLiteral("div3MaxBox"));
        div3MaxBox->setGeometry(QRect(400, 130, 41, 20));
        div1MinSlider = new QSlider(tab_6);
        div1MinSlider->setObjectName(QStringLiteral("div1MinSlider"));
        div1MinSlider->setGeometry(QRect(80, 50, 121, 19));
        div1MinSlider->setMaximum(100);
        div1MinSlider->setOrientation(Qt::Horizontal);
        div1MinSlider->setTickPosition(QSlider::TicksAbove);
        div2MinSlider = new QSlider(tab_6);
        div2MinSlider->setObjectName(QStringLiteral("div2MinSlider"));
        div2MinSlider->setGeometry(QRect(80, 90, 121, 19));
        div2MinSlider->setMaximum(100);
        div2MinSlider->setOrientation(Qt::Horizontal);
        div2MinSlider->setTickPosition(QSlider::TicksAbove);
        div3MinSlider = new QSlider(tab_6);
        div3MinSlider->setObjectName(QStringLiteral("div3MinSlider"));
        div3MinSlider->setGeometry(QRect(80, 130, 121, 19));
        div3MinSlider->setMaximum(100);
        div3MinSlider->setOrientation(Qt::Horizontal);
        div3MinSlider->setTickPosition(QSlider::TicksAbove);
        div4MinSlider = new QSlider(tab_6);
        div4MinSlider->setObjectName(QStringLiteral("div4MinSlider"));
        div4MinSlider->setGeometry(QRect(80, 170, 121, 19));
        div4MinSlider->setMaximum(100);
        div4MinSlider->setOrientation(Qt::Horizontal);
        div4MinSlider->setTickPosition(QSlider::TicksAbove);
        div1MaxSlider = new QSlider(tab_6);
        div1MaxSlider->setObjectName(QStringLiteral("div1MaxSlider"));
        div1MaxSlider->setGeometry(QRect(270, 50, 121, 19));
        div1MaxSlider->setMaximum(100);
        div1MaxSlider->setOrientation(Qt::Horizontal);
        div1MaxSlider->setTickPosition(QSlider::TicksAbove);
        div4MaxSlider = new QSlider(tab_6);
        div4MaxSlider->setObjectName(QStringLiteral("div4MaxSlider"));
        div4MaxSlider->setGeometry(QRect(270, 170, 121, 19));
        div4MaxSlider->setMaximum(100);
        div4MaxSlider->setOrientation(Qt::Horizontal);
        div4MaxSlider->setTickPosition(QSlider::TicksAbove);
        div2MaxSlider = new QSlider(tab_6);
        div2MaxSlider->setObjectName(QStringLiteral("div2MaxSlider"));
        div2MaxSlider->setGeometry(QRect(270, 90, 121, 19));
        div2MaxSlider->setMaximum(100);
        div2MaxSlider->setOrientation(Qt::Horizontal);
        div2MaxSlider->setTickPosition(QSlider::TicksAbove);
        div3MaxSlider = new QSlider(tab_6);
        div3MaxSlider->setObjectName(QStringLiteral("div3MaxSlider"));
        div3MaxSlider->setGeometry(QRect(270, 130, 121, 19));
        div3MaxSlider->setMaximum(100);
        div3MaxSlider->setOrientation(Qt::Horizontal);
        div3MaxSlider->setTickPosition(QSlider::TicksAbove);
        line = new QFrame(tab_6);
        line->setObjectName(QStringLiteral("line"));
        line->setGeometry(QRect(250, 50, 16, 141));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);
        label_10 = new QLabel(tab_6);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(10, 50, 46, 13));
        label_11 = new QLabel(tab_6);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setGeometry(QRect(10, 90, 46, 13));
        label_12 = new QLabel(tab_6);
        label_12->setObjectName(QStringLiteral("label_12"));
        label_12->setGeometry(QRect(10, 130, 46, 13));
        label_13 = new QLabel(tab_6);
        label_13->setObjectName(QStringLiteral("label_13"));
        label_13->setGeometry(QRect(10, 170, 46, 13));
        label_14 = new QLabel(tab_6);
        label_14->setObjectName(QStringLiteral("label_14"));
        label_14->setGeometry(QRect(120, 20, 46, 13));
        label_15 = new QLabel(tab_6);
        label_15->setObjectName(QStringLiteral("label_15"));
        label_15->setGeometry(QRect(310, 20, 46, 13));
        tabWidget->addTab(tab_6, QString());
        VolumeRendererClass->setCentralWidget(centralWidget);
        tabWidget->raise();
        pushButton->raise();
        menuBar = new QMenuBar(VolumeRendererClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 514, 21));
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
        QObject::connect(div1MinBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv1Min(QString)));
        QObject::connect(div1MaxBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv1Max(QString)));
        QObject::connect(div2MinBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv2Min(QString)));
        QObject::connect(div2MaxBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv2Max(QString)));
        QObject::connect(div3MinBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv3Min(QString)));
        QObject::connect(div3MaxBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv3Max(QString)));
        QObject::connect(div4MinBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv4Min(QString)));
        QObject::connect(div4MaxBox, SIGNAL(textEdited(QString)), VolumeRendererClass, SLOT(ChangeOpacityDiv4Max(QString)));
        QObject::connect(contourThresholdSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustContourThreshold(int)));
        QObject::connect(suggestiveThresholdSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustSuggestiveThreshold(int)));
        QObject::connect(numPixelsLowerSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustNumPixelsLower(int)));
        QObject::connect(kernelRadiusSlider, SIGNAL(valueChanged(int)), VolumeRendererClass, SLOT(AdjustKernelRadius(int)));
        QObject::connect(showDiffuseCheckbox, SIGNAL(toggled(bool)), VolumeRendererClass, SLOT(ToggleShowDiffuse(bool)));

        tabWidget->setCurrentIndex(0);
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
         << QApplication::translate("VolumeRendererClass", "X-Toon Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Raycast Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Lighting Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Opacity Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Depth Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Normals Shader", 0)
         << QApplication::translate("VolumeRendererClass", "Shadow Shader", 0)
        );
        shaderComboBox->setCurrentText(QApplication::translate("VolumeRendererClass", "X-Toon Shader", 0));
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
        label_10->setText(QApplication::translate("VolumeRendererClass", "Division 1", 0));
        label_11->setText(QApplication::translate("VolumeRendererClass", "Division 2", 0));
        label_12->setText(QApplication::translate("VolumeRendererClass", "Division 3", 0));
        label_13->setText(QApplication::translate("VolumeRendererClass", "Division 4", 0));
        label_14->setText(QApplication::translate("VolumeRendererClass", "Minimum", 0));
        label_15->setText(QApplication::translate("VolumeRendererClass", "Maximum", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_6), QApplication::translate("VolumeRendererClass", "Blank", 0));
    } // retranslateUi

};

namespace Ui {
    class VolumeRendererClass: public Ui_VolumeRendererClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VOLUMERENDERER_H
