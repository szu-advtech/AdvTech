#include "glwidget.h"


glwidget::glwidget(QWidget* parent) :QOpenGLWidget(parent)
{
    //setGeometry(0, 0, 800, 800);
    setFixedSize(800, 800);
}

void glwidget::initializeGL() {
    grid_num = 400 / grid_size;
    r = grid_size * sqrt(2);
    draw_disk_r = r;
    sampling_ = new Sampling();
    sampling_->grid_num = grid_num;
    sampling_->r = r;
    sampling_->grid_size = grid_size;

    sampling_->initilize();

    //sampling_->DartThrowing();
    //int miss_num = sampling_->miss_num;
    //int sample_num = sampling_->sample_num_1;
    //int try_num_1 = sampling_->try_num_1;
    //sampling_->computePolygonVoid();


    //bool flag_ = true;
    //while (flag_) {
    //    flag_ = sampling_->DartThrowingInVoid();
    //}
    //for (int i = 0; i < 10; i++) {
    //    sampling_->DartThrowingInVoid();
    //}

    
}

void glwidget::initLayout() {

    QVBoxLayout* layout = new QVBoxLayout;
    
}

void glwidget::resizeGL(int width, int height) {
    width_ = width;
    height_ = height;
}

void glwidget::paintGL() {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    drawGrids();
    drawPoints();
    drawDisks();
    drawVoids();
}

void glwidget::drawPoints() {
    glPointSize(2);  //1
    glBegin(GL_POINTS);
    for (int i = 0; i < grid_num; i++) {
        for (int j = 0; j < grid_num; j++) {
            if (sampling_->flag[i][j]) {
                double x = sampling_->sample_points[i][j].x;
                double y = sampling_->sample_points[i][j].y;
                if (x != -1 && y != -1) {
                    x = (x - 200) / 200;
                    y = (y - 200) / 200;
                    glColor3f(0.5f, 0.5f, 0.5f);
                    glVertex2f(x, y);
                }
            }
        }
    }
    glEnd();
}

void glwidget::drawDisks() {
    double disk_r = draw_disk_r / 200;
    float PI = 3.1415926f;
    glPointSize(0.5f);
    glBegin(GL_POINTS);
    for (int i = 0; i < grid_num; i++) {
        for (int j = 0; j < grid_num; j++) {
            if (sampling_->flag[i][j]) {
                double x = sampling_->sample_points[i][j].x;
                double y = sampling_->sample_points[i][j].y;
                if (x != -1 && y != -1) {
                    x = (x - 200) / 200;
                    y = (y - 200) / 200;    //center
                    for (int k = 0; k < 600; k++) {
                        glColor3f(0.5f, 0.5f, 0.5f);
                        glVertex2f(disk_r * cos(2 * PI * k / 600) + x,
                            disk_r * sin(2 * PI * k / 600) + y);
                    }
                }
            }
        }
    }
    glEnd();
    glFlush();
}

void glwidget::drawGrids() {
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    for (int i = 0; i < grid_num + 1; i++) {
        double bias = i * r / sqrt(2);
        bias = (bias - 200) / 200;
        glVertex2f(-1, bias);
        glVertex2f(1, bias);
        glVertex2f(bias, -1);
        glVertex2f(bias, 1);
    }
    glEnd();
    glFlush();
}

void glwidget::drawVoids() {
    if (sampling_->all_sample_num == 0 || !draw_void)
        return;
    for (int i = 0; i < grid_num; i++) {
        for (int j = 0; j < grid_num; j++) {
            if (sampling_->flag[i][j])
                continue;
            if (!sampling_->flag[i][j]) {
                std::vector<std::vector<std::pair<Sampling::point, int>>> polygons = sampling_->all_cells[i][j]->polygons;
                int polygon_num = polygons.size();
                for (int k = 0; k < polygon_num; k++) {
                    std::vector<std::pair<Sampling::point, int>> polygon1 = polygons[k];
                    int polygon_len = polygon1.size();
                    glBegin(GL_POLYGON);
                    for (int m = 0; m < polygon_len; m++) {
                        double x = polygon1[m].first.x;
                        double y = polygon1[m].first.y;
                        x = (x - 200) / 200;
                        y = (y - 200) / 200;
                        glColor3f(0.6f, 0.6f, 0.6f);
                        glVertex2f(x, y);
                    }
                    glEnd();
                }
            }
        }
    }
}

void glwidget::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) 
    {
    case Qt::Key_L:
        draw_disk_r = r;
        update();
        break;
    case Qt::Key_S:
        draw_disk_r = r / 2;
        update();
        break;
    case Qt::Key_E:
        sampling_->DartThrowingInVoid();
        update();
        break;
    }

}

void glwidget::update_grid_size(QString text) {
    bool ok;
    double value = text.toDouble(&ok);
    initializeGL();
    update();
}

void glwidget::start_SampleInGrid() {
    sampling_->DartThrowing();
    sampling_->computePolygonVoid();
    update();
}

void glwidget::start_SampleInVoid() {
    sampling_->DartThrowingInVoid();
    update();
}

void glwidget::half_DiskRadius() {
    draw_disk_r = r / 2;
    draw_void = false;
    update();
}

void glwidget::double_DiskRadius() {
    draw_disk_r = r;
    draw_void = true;
    update();
}

int glwidget::get_Sample_num() {
    return sampling_->all_sample_num;
}


