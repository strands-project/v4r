#pragma once

#include <QObject>
#include <QHBoxLayout>
#include <QWidget>
#include <QFileDialog>
#include <QLineEdit>

#include "manager.h"

class QParameterEdit : public QHBoxLayout
{
    Q_OBJECT

private:
    object_modeller::ParameterBase *param;
    QWidget *widget;

    void createWidget();

private slots:
    void parameterChanged(QString newValue)
    {
        param->fromString(newValue.toStdString());
    }
    void parameterChanged(bool newValue)
    {
        object_modeller::Parameter<bool> *cast = dynamic_cast<object_modeller::Parameter<bool> *>(param);
        *(cast->getValue()) = newValue;
    }
    void selectPath()
    {
        object_modeller::Parameter<std::string> *cast = dynamic_cast<object_modeller::Parameter<std::string> *>(param);
        QLineEdit *edit = dynamic_cast<QLineEdit *>(widget);

        QString dir = QString::fromStdString(*(cast->getValue()));
        QString folder = QFileDialog::getExistingDirectory(0, "Select Folder", dir);

        if (folder != NULL)
        {
            *(cast->getValue()) = folder.toUtf8().constData();
            edit->setText(folder);
        }
    }

public:
    virtual ~QParameterEdit()
    {
        while (this->count() > 0)
        {
            QLayoutItem *forDeletion = this->takeAt(0);

            if (forDeletion->widget())
            {
                forDeletion->widget()->deleteLater();
            }
        }
    }

    explicit QParameterEdit(object_modeller::ParameterBase *p, QWidget* parent=0) : QHBoxLayout(parent)
    {
        this->param = p;

        createWidget();
    }

    object_modeller::ParameterBase *getParameter()
    {
        return param;
    }
};
