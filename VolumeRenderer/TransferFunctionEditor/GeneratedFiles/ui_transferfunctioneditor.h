/********************************************************************************
** Form generated from reading UI file 'transferfunctioneditor.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRANSFERFUNCTIONEDITOR_H
#define UI_TRANSFERFUNCTIONEDITOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TransferFunctionEditor
{
public:
    QAction *action_Open_Transfer_Function;
    QAction *action_Save_Transfer_Function;
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QTabWidget *tabWidget;
    QWidget *tab;
    QHBoxLayout *horizontalLayout_5;
    QVBoxLayout *verticalLayout;
    QWidget *tab_2;
    QHBoxLayout *horizontalLayout_6;
    QVBoxLayout *verticalLayout_2;
    QWidget *tab_3;
    QHBoxLayout *horizontalLayout_7;
    QVBoxLayout *verticalLayout_3;
    QWidget *tab_4;
    QHBoxLayout *horizontalLayout_8;
    QVBoxLayout *verticalLayout_4;
    QWidget *tab_5;
    QHBoxLayout *horizontalLayout_9;
    QVBoxLayout *verticalLayout_5;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_0;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label;
    QDoubleSpinBox *doubleSpinBox;
    QPushButton *rampButton;
    QPushButton *flatButton;
    QPushButton *distributeHorizontallyButton;
    QPushButton *distributeVerticallyButton;
    QPushButton *diagonalButton;
    QPushButton *peaksButton;
    QHBoxLayout *horizontalLayout_4;
    QCheckBox *checkBox;
    QCheckBox *checkBox_2;
    QPushButton *computeDistanceButton;
    QPushButton *visibilityHistogramButton;
    QHBoxLayout *horizontalLayout_10;
    QPushButton *cameraButton;
    QPushButton *rotateButton;
    QPushButton *frontButton;
    QPushButton *leftButton;
    QPushButton *topButton;
    QHBoxLayout *horizontalLayout_2;
    QSpinBox *spinBox;
    QPushButton *entropyButton;
    QPushButton *visibilityButton;

    void setupUi(QMainWindow *TransferFunctionEditor)
    {
        if (TransferFunctionEditor->objectName().isEmpty())
            TransferFunctionEditor->setObjectName(QStringLiteral("TransferFunctionEditor"));
        TransferFunctionEditor->resize(728, 480);
        action_Open_Transfer_Function = new QAction(TransferFunctionEditor);
        action_Open_Transfer_Function->setObjectName(QStringLiteral("action_Open_Transfer_Function"));
        action_Save_Transfer_Function = new QAction(TransferFunctionEditor);
        action_Save_Transfer_Function->setObjectName(QStringLiteral("action_Save_Transfer_Function"));
        centralWidget = new QWidget(TransferFunctionEditor);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        horizontalLayout_5 = new QHBoxLayout(tab);
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);

        horizontalLayout_5->addLayout(verticalLayout);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        horizontalLayout_6 = new QHBoxLayout(tab_2);
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));

        horizontalLayout_6->addLayout(verticalLayout_2);

        tabWidget->addTab(tab_2, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        horizontalLayout_7 = new QHBoxLayout(tab_3);
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));

        horizontalLayout_7->addLayout(verticalLayout_3);

        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QStringLiteral("tab_4"));
        horizontalLayout_8 = new QHBoxLayout(tab_4);
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        horizontalLayout_8->setContentsMargins(0, 0, 0, 0);
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));

        horizontalLayout_8->addLayout(verticalLayout_4);

        tabWidget->addTab(tab_4, QString());
        tab_5 = new QWidget();
        tab_5->setObjectName(QStringLiteral("tab_5"));
        horizontalLayout_9 = new QHBoxLayout(tab_5);
        horizontalLayout_9->setSpacing(6);
        horizontalLayout_9->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        horizontalLayout_9->setContentsMargins(0, 0, 0, 0);
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));

        horizontalLayout_9->addLayout(verticalLayout_5);

        tabWidget->addTab(tab_5, QString());

        horizontalLayout->addWidget(tabWidget);

        TransferFunctionEditor->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(TransferFunctionEditor);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 728, 26));
        menu_File = new QMenu(menuBar);
        menu_File->setObjectName(QStringLiteral("menu_File"));
        TransferFunctionEditor->setMenuBar(menuBar);
        mainToolBar = new QToolBar(TransferFunctionEditor);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        TransferFunctionEditor->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(TransferFunctionEditor);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        TransferFunctionEditor->setStatusBar(statusBar);
        dockWidget = new QDockWidget(TransferFunctionEditor);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        verticalLayout_0 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_0->setSpacing(6);
        verticalLayout_0->setContentsMargins(11, 11, 11, 11);
        verticalLayout_0->setObjectName(QStringLiteral("verticalLayout_0"));
        verticalLayout_0->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(dockWidgetContents);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout_3->addWidget(label);

        doubleSpinBox = new QDoubleSpinBox(dockWidgetContents);
        doubleSpinBox->setObjectName(QStringLiteral("doubleSpinBox"));
        doubleSpinBox->setDecimals(2);
        doubleSpinBox->setMaximum(1);
        doubleSpinBox->setSingleStep(0.01);
        doubleSpinBox->setValue(0.1);

        horizontalLayout_3->addWidget(doubleSpinBox);

        rampButton = new QPushButton(dockWidgetContents);
        rampButton->setObjectName(QStringLiteral("rampButton"));

        horizontalLayout_3->addWidget(rampButton);

        flatButton = new QPushButton(dockWidgetContents);
        flatButton->setObjectName(QStringLiteral("flatButton"));

        horizontalLayout_3->addWidget(flatButton);

        distributeHorizontallyButton = new QPushButton(dockWidgetContents);
        distributeHorizontallyButton->setObjectName(QStringLiteral("distributeHorizontallyButton"));

        horizontalLayout_3->addWidget(distributeHorizontallyButton);

        distributeVerticallyButton = new QPushButton(dockWidgetContents);
        distributeVerticallyButton->setObjectName(QStringLiteral("distributeVerticallyButton"));

        horizontalLayout_3->addWidget(distributeVerticallyButton);

        diagonalButton = new QPushButton(dockWidgetContents);
        diagonalButton->setObjectName(QStringLiteral("diagonalButton"));

        horizontalLayout_3->addWidget(diagonalButton);

        peaksButton = new QPushButton(dockWidgetContents);
        peaksButton->setObjectName(QStringLiteral("peaksButton"));

        horizontalLayout_3->addWidget(peaksButton);


        verticalLayout_0->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        checkBox = new QCheckBox(dockWidgetContents);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        checkBox->setChecked(false);

        horizontalLayout_4->addWidget(checkBox);

        checkBox_2 = new QCheckBox(dockWidgetContents);
        checkBox_2->setObjectName(QStringLiteral("checkBox_2"));

        horizontalLayout_4->addWidget(checkBox_2);

        computeDistanceButton = new QPushButton(dockWidgetContents);
        computeDistanceButton->setObjectName(QStringLiteral("computeDistanceButton"));

        horizontalLayout_4->addWidget(computeDistanceButton);

        visibilityHistogramButton = new QPushButton(dockWidgetContents);
        visibilityHistogramButton->setObjectName(QStringLiteral("visibilityHistogramButton"));

        horizontalLayout_4->addWidget(visibilityHistogramButton);


        verticalLayout_0->addLayout(horizontalLayout_4);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setSpacing(6);
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        horizontalLayout_10->setContentsMargins(7, 7, 7, 7);
        cameraButton = new QPushButton(dockWidgetContents);
        cameraButton->setObjectName(QStringLiteral("cameraButton"));

        horizontalLayout_10->addWidget(cameraButton);

        rotateButton = new QPushButton(dockWidgetContents);
        rotateButton->setObjectName(QStringLiteral("rotateButton"));

        horizontalLayout_10->addWidget(rotateButton);

        frontButton = new QPushButton(dockWidgetContents);
        frontButton->setObjectName(QStringLiteral("frontButton"));

        horizontalLayout_10->addWidget(frontButton);

        leftButton = new QPushButton(dockWidgetContents);
        leftButton->setObjectName(QStringLiteral("leftButton"));

        horizontalLayout_10->addWidget(leftButton);

        topButton = new QPushButton(dockWidgetContents);
        topButton->setObjectName(QStringLiteral("topButton"));

        horizontalLayout_10->addWidget(topButton);


        verticalLayout_0->addLayout(horizontalLayout_10);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, -1);
        spinBox = new QSpinBox(dockWidgetContents);
        spinBox->setObjectName(QStringLiteral("spinBox"));
        spinBox->setMinimum(1);
        spinBox->setMaximum(65536);
        spinBox->setValue(1000);

        horizontalLayout_2->addWidget(spinBox);

        entropyButton = new QPushButton(dockWidgetContents);
        entropyButton->setObjectName(QStringLiteral("entropyButton"));

        horizontalLayout_2->addWidget(entropyButton);

        visibilityButton = new QPushButton(dockWidgetContents);
        visibilityButton->setObjectName(QStringLiteral("visibilityButton"));

        horizontalLayout_2->addWidget(visibilityButton);


        verticalLayout_0->addLayout(horizontalLayout_2);

        dockWidget->setWidget(dockWidgetContents);
        TransferFunctionEditor->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dockWidget);

        menuBar->addAction(menu_File->menuAction());
        menu_File->addAction(action_Open_Transfer_Function);
        menu_File->addAction(action_Save_Transfer_Function);

        retranslateUi(TransferFunctionEditor);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(TransferFunctionEditor);
    } // setupUi

    void retranslateUi(QMainWindow *TransferFunctionEditor)
    {
        TransferFunctionEditor->setWindowTitle(QApplication::translate("TransferFunctionEditor", "Transfer Function Editor", Q_NULLPTR));
        action_Open_Transfer_Function->setText(QApplication::translate("TransferFunctionEditor", "&Open Transfer Function...", Q_NULLPTR));
        action_Save_Transfer_Function->setText(QApplication::translate("TransferFunctionEditor", "&Save Transfer Function...", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("TransferFunctionEditor", "transfer function", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("TransferFunctionEditor", "intensity histogram", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("TransferFunctionEditor", "visibility histogram", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QApplication::translate("TransferFunctionEditor", "frustum", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab_5), QApplication::translate("TransferFunctionEditor", "xtoon", Q_NULLPTR));
        menu_File->setTitle(QApplication::translate("TransferFunctionEditor", "&File", Q_NULLPTR));
        label->setText(QApplication::translate("TransferFunctionEditor", "opacity", Q_NULLPTR));
        rampButton->setText(QApplication::translate("TransferFunctionEditor", "ramp", Q_NULLPTR));
        flatButton->setText(QApplication::translate("TransferFunctionEditor", "flat", Q_NULLPTR));
        distributeHorizontallyButton->setText(QApplication::translate("TransferFunctionEditor", "horizontal", Q_NULLPTR));
        distributeVerticallyButton->setText(QApplication::translate("TransferFunctionEditor", "vertical", Q_NULLPTR));
        diagonalButton->setText(QApplication::translate("TransferFunctionEditor", "diagonal", Q_NULLPTR));
        peaksButton->setText(QApplication::translate("TransferFunctionEditor", "peaks", Q_NULLPTR));
        checkBox->setText(QApplication::translate("TransferFunctionEditor", "Ma's Optimizer", Q_NULLPTR));
        checkBox_2->setText(QApplication::translate("TransferFunctionEditor", "Luo's Optimizer", Q_NULLPTR));
        computeDistanceButton->setText(QApplication::translate("TransferFunctionEditor", "compute distance", Q_NULLPTR));
        visibilityHistogramButton->setText(QApplication::translate("TransferFunctionEditor", "visibility histogram", Q_NULLPTR));
        cameraButton->setText(QApplication::translate("TransferFunctionEditor", "save camera", Q_NULLPTR));
        rotateButton->setText(QApplication::translate("TransferFunctionEditor", "rotate", Q_NULLPTR));
        frontButton->setText(QApplication::translate("TransferFunctionEditor", "front", Q_NULLPTR));
        leftButton->setText(QApplication::translate("TransferFunctionEditor", "left", Q_NULLPTR));
        topButton->setText(QApplication::translate("TransferFunctionEditor", "top", Q_NULLPTR));
        entropyButton->setText(QApplication::translate("TransferFunctionEditor", "entropy", Q_NULLPTR));
        visibilityButton->setText(QApplication::translate("TransferFunctionEditor", "visibility", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class TransferFunctionEditor: public Ui_TransferFunctionEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRANSFERFUNCTIONEDITOR_H
