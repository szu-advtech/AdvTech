#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include "Sampling.h"
#include <QKeyEvent>
#include <QLineEdit>
#include <QVBoxLayout>

class glwidget :public QOpenGLWidget
{
	Q_OBJECT
public:
    glwidget(QWidget* parent = 0);
    int get_Sample_num();

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void drawPoints();
    void drawDisks();
    void drawGrids();
    void drawVoids();
    void keyPressEvent(QKeyEvent *event);
    void initLayout();

public:
    Sampling* sampling_;
    double grid_size = 16;
    int grid_num = 400 / grid_size;
    double r = grid_size * sqrt(2);
    int width_;
    int height_;

public slots:
    void update_grid_size(QString text);
    void start_SampleInGrid();
    void start_SampleInVoid();
    void half_DiskRadius();
    void double_DiskRadius();
    
private:
    double draw_disk_r = r;
    bool draw_void = true;
};


#endif //GLWIDGET_H

