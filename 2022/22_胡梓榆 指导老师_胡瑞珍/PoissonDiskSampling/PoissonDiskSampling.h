#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_PoissonDiskSampling.h"
#include "Sampling.h"
#include "glwidget.h"
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QFont>

class PoissonDiskSampling : public QMainWindow
{
    Q_OBJECT

public:
    PoissonDiskSampling(QWidget *parent = Q_NULLPTR);

public:
    glwidget *glwidget_;

private:
    Ui::PoissonDiskSamplingClass ui;
    QLineEdit* m_grid_size_Edit;
    QLabel* m_sample_num_Label;
    QLineEdit* m_sample_num_Edit;
    QPushButton* startSampleInGridBtn;
    QPushButton* startSampleInVoidBtn;
    QPushButton* halfDiskRadiusBtn;
    QPushButton* doubleDiskRadiusBtn;
    void init();
    void initLayout();
    QGroupBox* CreateParameterGroup();

private slots:
    void pass_grid_sizeText(QString text);
    void show_sample_num();
};
