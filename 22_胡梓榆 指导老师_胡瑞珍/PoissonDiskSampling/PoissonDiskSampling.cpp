#include "PoissonDiskSampling.h"
#include <QFileDialog>
#include <QString>
#include <QtGui>
#include <QtOpenGL>
#include <QtOpenGL/qgl.h>

#include <stdlib.h>
#include <gl/GL.h>
#include <gl/GLU.h>

PoissonDiskSampling::PoissonDiskSampling(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setWindowTitle("PoissonDiskSampling");
    setGeometry(400, 100, 1000, 800);
    init();
    initLayout();
    connect(m_grid_size_Edit, &QLineEdit::textChanged, this, &PoissonDiskSampling::pass_grid_sizeText);
    connect(m_grid_size_Edit, &QLineEdit::textChanged, glwidget_, &glwidget::update_grid_size);

    connect(startSampleInGridBtn, &QPushButton::clicked, glwidget_, &glwidget::start_SampleInGrid);
    connect(startSampleInGridBtn, &QPushButton::clicked, this, &PoissonDiskSampling::show_sample_num);
    connect(startSampleInVoidBtn, &QPushButton::clicked, glwidget_, &glwidget::start_SampleInVoid);
    connect(startSampleInVoidBtn, &QPushButton::clicked, this, &PoissonDiskSampling::show_sample_num);

    connect(halfDiskRadiusBtn, &QPushButton::clicked, glwidget_, &glwidget::half_DiskRadius);
    connect(doubleDiskRadiusBtn, &QPushButton::clicked, glwidget_, &glwidget::double_DiskRadius);

}

void PoissonDiskSampling::init() {
    glwidget_ = new glwidget();
}

void PoissonDiskSampling::initLayout() {
    QHBoxLayout* layout_main = new QHBoxLayout;
    centralWidget()->setLayout(layout_main);

    glwidget_->setFixedSize(800, 800);
    layout_main->addWidget(glwidget_);
    layout_main->addWidget(CreateParameterGroup());
}

QGroupBox *PoissonDiskSampling::CreateParameterGroup() {
    QFont font("Microsoft YaHei", 18, 50);
    m_grid_size_Edit = new QLineEdit();
    m_grid_size_Edit->setFont(font);
    //m_sample_num_Edit = new QLineEdit();
    
    startSampleInGridBtn = new QPushButton(tr("Start Sample in Grid"));
    startSampleInVoidBtn = new QPushButton(tr("Start Sample in Void"));
    halfDiskRadiusBtn = new QPushButton(tr("half Disk Radius"));
    doubleDiskRadiusBtn = new QPushButton(tr("double Disk Radius"));
    
    m_sample_num_Label = new QLabel();
    m_sample_num_Label->setFont(font);
    QGroupBox* groupBox = new QGroupBox(tr("Input Parameter"));

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(new QLabel(tr("#Grid size:")));
    layout->addWidget(m_grid_size_Edit);
    layout->addWidget(startSampleInGridBtn);
    layout->addWidget(startSampleInVoidBtn);
    layout->addWidget(halfDiskRadiusBtn);
    layout->addWidget(doubleDiskRadiusBtn);
    layout->addWidget(new QLabel(tr("#Sample num:")));
    layout->addWidget(m_sample_num_Label);

    //layout->addWidget(m_sample_num_Edit);

    //QGridLayout* gridLayout = new QGridLayout;
    //gridLayout->addWidget(new QLabel(tr("#Grid size:")), 0, 0, 1, 1);
    //gridLayout->addWidget(m_grid_size_Edit, 0, 1, 1, 2);
    //gridLayout->addWidget(new QLabel(tr("#Sample num:")), 1, 0, 1, 1);
    //gridLayout->addWidget(m_sample_num_Edit, 1, 1, 1, 2);


    groupBox->setLayout(layout);
    return groupBox;
}

void PoissonDiskSampling::pass_grid_sizeText(QString text) {
    bool ok;
    double value = text.toDouble(&ok);
    if (ok) {
        glwidget_->grid_size = value;
    }
}

void PoissonDiskSampling::show_sample_num() {
    int sample_num = glwidget_->get_Sample_num();
    QString text = QString::number(sample_num);
    m_sample_num_Label->setText(text);
}
