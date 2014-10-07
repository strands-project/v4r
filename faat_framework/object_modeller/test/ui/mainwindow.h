#pragma once

#include <QtGui/QMainWindow>

#include <QTimer>

#include "settings.h"

class QCloseEvent;
class Manager;
class Renderer;

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit MainWindow(Manager* manager, QWidget *parent = 0);
  ~MainWindow();

    void loop();

    void closeEvent (QCloseEvent *event);

    void updateRenderControls();
    void updateRenderControlsImpl();

    void updateRenderer();

    void updateImage();

signals:
public slots:
   void doUpdateRenderer();

private slots:
    void on_actionSettings_triggered();

    void on_btnProcess_clicked();

    void on_btnStep_clicked();

    void on_btnPause_clicked();

    void on_btnReset_clicked();

    void on_cmbSequence_currentIndexChanged(int index);

    void on_btnObjectPrevious_clicked();

    void on_btnObjectNext_clicked();

    void on_btnSnapshot_clicked();

    void on_btnContinue_clicked();

    void on_btnAddSequence_clicked();

    void on_btnSave_clicked();

    void on_btnStartRecording_clicked();

    void on_btnStopRecording_clicked();

private:
    bool m_updateRenderer;
    bool m_updateImage;
    bool m_updateRenderControls;
    object_modeller::output::Renderer::Ptr renderer;
    Manager *manager;

    QTimer * timer;
    boost::thread *process_thread;
    Ui::MainWindow *m_window;
    Settings *m_settings;

};
