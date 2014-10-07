#include "settings.h"
#include "ui_settings.h"

#include <QLineEdit>
#include <QComboBox>
#include <QFileDialog>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QVBoxLayout>

//Q_DECLARE_METATYPE(InputModule*)

#include "parameteredit.h"

Settings::Settings(Manager *manager, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Settings),
    manager(manager)
{
    ui->setupUi(this);

    currentConfig = manager->getConfig();
}

void Settings::initPipeline()
{
    for (int i=0;i<manager->getPipeline()->getSteps().size();i++)
    {
        object_modeller::PipelineStep *step = manager->getPipeline()->getSteps().at(i);

        QString itemString = QString::number(i);
        itemString.append(" ");
        itemString.append(QString::fromStdString(step->getName()));

        QTreeWidgetItem *item = new QTreeWidgetItem(QStringList(itemString));

        object_modeller::PipelineStepMulti *multi = dynamic_cast<object_modeller::PipelineStepMulti*>(step);

        if (multi != NULL)
        {
            for (int j=0;j<multi->getSteps().size();j++)
            {
                step = multi->getSteps().at(j);

                QString subItemString = QString::number(j);
                subItemString.append(" ");
                subItemString.append(QString::fromStdString(step->getName()));

                QTreeWidgetItem *subItem = new QTreeWidgetItem(QStringList(subItemString));
                item->addChild(subItem);
            }
        }

        ui->pipelineList->addTopLevelItem(item);

        //QString key = QString::fromStdString(manager->getInputModules().at(i)->getName());
        //QVariant v = qVariantFromValue((void*) manager->getInputModules().at(i));
        //ui->cmbInput->addItem(key, v);
    }
}

Settings::~Settings()
{
    delete ui;
}

void QParameterEdit::createWidget()
{
    if (param->getType() == object_modeller::ParameterBase::BOOL)
    {
        QCheckBox *checkbox = new QCheckBox();

        object_modeller::Parameter<bool> *cast = dynamic_cast<object_modeller::Parameter<bool> *>(param);
        checkbox->setChecked(*(cast->getValue()));

        connect(checkbox, SIGNAL(toggled(bool)), this, SLOT(parameterChanged(bool)));

        widget = checkbox;
    }
    else if (param->getType() == object_modeller::ParameterBase::INT)
    {
        QSpinBox *spinbox = new QSpinBox();

        object_modeller::Parameter<int> *cast = dynamic_cast<object_modeller::Parameter<int> *>(param);
        spinbox->setMaximum(INT_MAX);
        spinbox->setValue(*cast->getValue());

        connect(spinbox, SIGNAL(valueChanged(QString)), this, SLOT(parameterChanged(QString)));

        widget = spinbox;
    }
    else if (param->getType() == object_modeller::ParameterBase::FLOAT)
    {
        QDoubleSpinBox *spinbox = new QDoubleSpinBox();
        spinbox->setLocale(QLocale(QLocale::English));
        spinbox->setDecimals(5);

        object_modeller::Parameter<float> *cast = dynamic_cast<object_modeller::Parameter<float> *>(param);
        spinbox->setValue(*cast->getValue());

        connect(spinbox, SIGNAL(valueChanged(QString)), this, SLOT(parameterChanged(QString)));

        widget = spinbox;
    }
    else if (param->getType() == object_modeller::ParameterBase::FOLDER)
    {
        QLineEdit *line = new QLineEdit();
        line->setText(QString::fromStdString(param->toString()));

        connect(line, SIGNAL(textChanged(QString)), this, SLOT(parameterChanged(QString)));

        widget = line;

        QPushButton *btn = new QPushButton();
        btn->setText("...");

        connect(btn, SIGNAL(clicked()), this, SLOT(selectPath()));

        this->addWidget(btn);
    }
    else
    {
        QLineEdit *line = new QLineEdit();
        line->setText(QString::fromStdString(param->toString()));

        connect(line, SIGNAL(textChanged(QString)), this, SLOT(parameterChanged(QString)));

        widget = line;
    }

    this->addWidget(widget);
}

void Settings::reloadParams(bool reloadAll)
{
    if (ui->pipelineList->currentIndex().row() < 0)
    {
        return;
    }

    std::cout << "reload params" << std::endl;

    int deletionOffset = 2;

    if (reloadAll)
    {
        deletionOffset = 0;
    }

    while (ui->formLayout->count() > deletionOffset)
    {
        QLayoutItem *forDeletion = ui->formLayout->takeAt(deletionOffset);

        if (forDeletion->widget())
        {
            QComboBox *combo = dynamic_cast<QComboBox*>(forDeletion->widget());

            if (combo != NULL)
            {
                disconnect(combo, SIGNAL(currentIndexChanged(int)), this, SLOT(on_combo_changed(int)));
            }

            forDeletion->widget()->deleteLater();
        }

        if (forDeletion->layout())
        {
            std::cout << "--------------------delete layout for " << forDeletion->layout() << std::endl;
            forDeletion->layout()->deleteLater();
        }

        //ui->formLayout->removeWidget(forDeletion->widget());
        //delete forDeletion->widget();
        //delete forDeletion;
    }

    this->update();


    object_modeller::PipelineStep *step = NULL;

    int index = ui->pipelineList->currentIndex().row();

    if (ui->pipelineList->currentIndex().parent().isValid() == false)
    {
        std::cout << "selected parent at " << index << std::endl;
        step = manager->getPipeline()->getSteps().at(index);
    }
    else
    {
        std::cout << "selected child at " << index << std::endl;
        int parentIndex = ui->pipelineList->currentIndex().parent().row();

        object_modeller::PipelineStepMulti *multi = dynamic_cast<object_modeller::PipelineStepMulti*>(manager->getPipeline()->getSteps().at(parentIndex));

        step = multi->getSteps().at(index);
    }

    std::cout << "get algorithm" << std::endl;

    // algorithm selection
    if (reloadAll)
    {
        object_modeller::PipelineStepChoiceBase *choice = dynamic_cast<object_modeller::PipelineStepChoiceBase*>(step);
        if (choice != NULL)
        {
            QComboBox *combo = new QComboBox();

            for (int i=0;i<choice->getSteps().size();i++)
            {
                QString name = QString::fromStdString(choice->getSteps().at(i)->getName());
                combo->addItem(name);
            }

            combo->setCurrentIndex(choice->getActiveChoice());

            connect(combo, SIGNAL(currentIndexChanged(int)), this, SLOT(comboChanged(int)));

            ui->formLayout->addRow(QString("Algorithm"), combo);
        }
    }

    std::cout << "get params" << std::endl;

    // parameters
    for (int j=0;j<step->getParameters().size();j++)
    {
        object_modeller::ParameterBase *p = step->getParameters().at(j);

        std::cout << "added row for " << p->getName() << std::endl;

        QString name = QString::fromStdString(p->getName());

        QParameterEdit *line = new QParameterEdit(p);

        ui->formLayout->addRow(name, line);
    }

    std::cout << "finished" << std::endl;

}

void Settings::accept()
{
    manager->applyParametersToConfig(manager->getConfig());
    manager->applyConfig(manager->getConfig());
    manager->getConfig()->save();
    reloadParams(true);
    QDialog::accept();
}


void Settings::reject()
{
    manager->applyConfig(manager->getConfig());
    reloadParams(true);
    QDialog::reject();
}

void Settings::on_pipelineList_itemSelectionChanged()
{
    std::cout << "item changed" << std::endl;

    reloadParams(true);
}

void Settings::comboChanged(int index)
{
    std::cout << "algorithm changed" << std::endl;

    int pipelineIndex = ui->pipelineList->currentIndex().row();

    std::cout << "index: " << pipelineIndex << " - " << index << std::endl;

    object_modeller::PipelineStepChoiceBase *choice = dynamic_cast<object_modeller::PipelineStepChoiceBase*>(manager->getPipeline()->getSteps().at(pipelineIndex));

    std::cout << "set active choice" << std::endl;

    choice->setActiveChoice(index);

    reloadParams(false);

    std::cout << "complete" << std::endl;
}

void Settings::on_btnImport_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Configuration"),
                                                     "",
                                                     tr("Configuration Files (*.txt)"));

    if (fileName != NULL)
    {
        std::cout << "Load config " << fileName.toUtf8().constData() << std::endl;
        currentConfig.reset(new object_modeller::Config());
        currentConfig->loadFromFile(fileName.toUtf8().constData());

        manager->applyConfig(currentConfig);

        reloadParams(true);
    }
}

void Settings::on_btnExport_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Configuration"),
                                                     "",
                                                     tr("Configuration Files (*.txt)"));

    if (fileName != NULL)
    {
        std::cout << "Save config " << fileName.toUtf8().constData() << std::endl;
        object_modeller::Config::Ptr c(new object_modeller::Config());
        manager->applyParametersToConfig(c);

        c->saveToFile(fileName.toUtf8().constData());
    }
}
