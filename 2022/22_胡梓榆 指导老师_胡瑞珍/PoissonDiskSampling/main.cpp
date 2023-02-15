#include "PoissonDiskSampling.h"
#include <QtWidgets/QApplication>
#include "glwidget.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //glwidget w;
    PoissonDiskSampling w;
    w.show();
    return a.exec();
}
