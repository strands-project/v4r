#ifndef SETTINGS_H
#define SETTINGS_H

#include <QDialog>

#include "manager.h"

//#include "ui_settings.h"

namespace Ui {
    class Settings;
}

class Settings : public QDialog
{
    Q_OBJECT
    
public:
    explicit Settings(Manager *manager, QWidget *parent = 0);
    ~Settings();

    void initPipeline();
    
private slots:

    void on_pipelineList_itemSelectionChanged();
    void comboChanged(int index);

    void on_btnImport_clicked();

    void on_btnExport_clicked();

private:
    Ui::Settings *ui;
    Manager *manager;

    object_modeller::Config::Ptr currentConfig;

    void reloadParams(bool reloadAll);
    void accept();
    void reject();

};

#endif // SETTINGS_H
