
#include "mainwindow.h"
#include <QApplication>

#include <clocale>

int main(int argc, char *argv[])
{
    Manager *m = new Manager();

    QApplication app(argc, argv);
    std::setlocale(LC_ALL,"C");

    MainWindow window(m);
    window.show();

  return app.exec();
}
