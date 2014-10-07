#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QCloseEvent>

#include "glrenderer.h"

MainWindow::MainWindow(Manager *manager, QWidget *parent) : QMainWindow(parent), m_window(new Ui::MainWindow), m_settings(new Settings(manager, this))
{
    this->manager = manager;

    m_updateRenderer = false;
    m_updateImage = false;
    m_updateRenderControls = false;

  m_window->setupUi(this);

  setWindowTitle(tr("Object Modeller UI"));

  renderer.reset(new QPclRenderer(this, m_window->lblImage, m_window->qvtkWidget));
  this->manager->setupPipeline(renderer);
  m_settings->initPipeline();

  m_window->progressBar->setVisible(false);
  m_window->lblStepName->setText(QString(""));
  m_window->lblImage->setVisible(false);

  /*
  m_window->lblStepName->setVisible(false);
  m_window->progressBar->setVisible(false);
  m_window->groupRenderer->setVisible(false);
  m_window->btnSnapshot->setVisible(false);
  m_window->btnContinue->setVisible(false);
  m_window->btnSave->setVisible(false);
  m_window->btnAddSequence->setVisible(false);
  */

  updateRenderControlsImpl();

  timer = new QTimer();
  QObject::connect(timer,SIGNAL(timeout()),this,SLOT(doUpdateRenderer()));
  timer->start(100);

  process_thread = new boost::thread(boost::bind(&MainWindow::loop, this));
}

void MainWindow::doUpdateRenderer()
{
    if (m_updateRenderer)
    {
        std::cout << "update from timer" << std::endl;
        std::cout << "update qt renderer" << std::endl;
        m_updateRenderer = false;

        if (renderer->hasImage())
        {
            m_window->lblImage->setVisible(true);
        }
        else
        {
            m_window->lblImage->setVisible(false);
        }

        updateRenderControlsImpl();
        renderer->update();

        /*
        if (pipeline_state == EventManager::NEW_FRAME)
        {
            m_window->btnSnapshot->setVisible(true);
            m_window->btnContinue->setVisible(true);
            m_window->groupRenderer->setEnabled(true);
            updateRenderControls();
            renderer->update();
            return;
        }
        else
        {
            m_window->btnSnapshot->setVisible(false);
            m_window->btnContinue->setVisible(false);
        }

        bool ur = false;

        if (pipeline_state == EventManager::INITIALIZED)
        {
            ur = true;

            m_window->actionSettings->setEnabled(true);
            m_window->btnProcess->setEnabled(true);
            m_window->btnStep->setEnabled(true);
            m_window->btnPause->setEnabled(false);
            m_window->btnReset->setEnabled(false);
            m_window->progressBar->setVisible(false);
            m_window->lblStepName->setVisible(false);
            m_window->groupRenderer->setVisible(false);
        }

        if (pipeline_state == EventManager::RUNNING)
        {
            m_window->actionSettings->setEnabled(false);
            m_window->btnProcess->setEnabled(false);
            m_window->btnStep->setEnabled(false);
            m_window->btnPause->setEnabled(true);
            m_window->btnReset->setEnabled(false);
            m_window->progressBar->setVisible(true);
            m_window->lblStepName->setVisible(true);
            m_window->groupRenderer->setEnabled(false);
            m_window->btnSave->setVisible(false);
        }

        if (pipeline_state == EventManager::PAUSED)
        {
            ur = true;

            m_window->actionSettings->setEnabled(false);
            m_window->btnProcess->setEnabled(true);
            m_window->btnStep->setEnabled(true);
            m_window->btnPause->setEnabled(false);
            m_window->btnReset->setEnabled(true);
            m_window->progressBar->setVisible(false);
            m_window->lblStepName->setVisible(false);
            m_window->groupRenderer->setEnabled(true);

        }

        if (pipeline_state == EventManager::CONFIGURE)
        {
            ur = true;

            m_window->groupRenderer->setEnabled(true);
            m_window->btnContinue->setVisible(true);
            m_window->btnSave->setVisible(true);
        }
        */
    }

    if (m_updateImage)
    {
        std::cout << "update from timer" << std::endl;
        m_updateImage = false;

        renderer->updateImage();
    }

    if (m_updateRenderControls)
    {
        std::cout << "update from timer" << std::endl;
        m_updateRenderControls = false;

        updateRenderControlsImpl();
    }
}

void MainWindow::updateRenderControls()
{
    m_updateRenderControls = true;
}

void MainWindow::updateRenderControlsImpl()
{
    std::cout << "update render controls" << std::endl;
    // update output text display
    m_window->lblRenderName->setText(QString::fromStdString(renderer->getRenderName()));

    /*
    if (renderer->getNrSequences() == 0)
    {
        m_window->groupRenderer->setVisible(false);
    }
    else
    {
        m_window->groupRenderer->setVisible(true);
    }
    */


    if (manager->getPipeline()->getState() == EventManager::RUNNING)
    {
        m_window->lblStepName->setVisible(true);
        m_window->progressBar->setVisible(true);
    }
    else
    {
        m_window->lblStepName->setVisible(false);
        m_window->progressBar->setVisible(false);
    }

    // update sequence combo box
    int nrSequences = renderer->getNrSequences();
    int sequenceId = renderer->getActiveSequenceId();

    bool oldState = m_window->cmbSequence->blockSignals(true);
    m_window->cmbSequence->clear();

    for (int i=0;i<nrSequences;i++)
    {
        QString itemText("Sequence ");
        itemText.append(QString::number(i));
        m_window->cmbSequence->addItem(itemText);
    }

    m_window->cmbSequence->setCurrentIndex(sequenceId);

    m_window->cmbSequence->blockSignals(oldState);

    // update object selection text
    int objectId = renderer->getActiveObjectId();
    int nrObjects = renderer->getNrObjects(sequenceId);

    if (objectId == -1)
    {
        m_window->lblObject->setText(QString("All Objects"));
    }
    else
    {
        QString currentObject("Object ");
        currentObject.append(QString::number(objectId + 1));
        currentObject.append("/");
        currentObject.append(QString::number(nrObjects));
        m_window->lblObject->setText(currentObject);
    }

    // update processed module
    if (manager->getPipeline()->getActiveStepName() != "")
    {
        QString text("Processing module: ");
        text.append(QString::fromStdString(manager->getPipeline()->getActiveStepName()));
        m_window->lblStepName->setText(text);
    }

    // update button states
    m_window->btnStep->setEnabled( renderer->isEventAvailable(EventManager::STEP) );
    m_window->btnStartRecording->setVisible( renderer->isEventAvailable(EventManager::START_RECORDING) );
    m_window->btnStopRecording->setVisible( renderer->isEventAvailable(EventManager::END_RECORDING) );
    m_window->btnContinue->setVisible( renderer->isEventAvailable(EventManager::CONTINUE) );
    m_window->btnSnapshot->setVisible( renderer->isEventAvailable(EventManager::GRAB_FRAME) );
    m_window->btnAddSequence->setVisible( renderer->isEventAvailable(EventManager::ADD_SEQUENCE) );
    m_window->btnSave->setVisible( renderer->isEventAvailable(EventManager::SAVE_CONFIG) );
    m_window->btnProcess->setEnabled( renderer->isEventAvailable(EventManager::RUN) );
    m_window->btnPause->setEnabled( renderer->isEventAvailable(EventManager::PAUSE) );
    m_window->btnReset->setEnabled( renderer->isEventAvailable(EventManager::RESET) );
    m_window->btnObjectNext->setEnabled( renderer->isEventAvailable(EventManager::NEXT_OBJECT) );
    m_window->btnObjectPrevious->setEnabled( renderer->isEventAvailable(EventManager::PREVIOUS_OBJECT) );
    m_window->cmbSequence->setEnabled( renderer->isEventAvailable(EventManager::PREVIOUS_SEQUENCE) || renderer->isEventAvailable(EventManager::NEXT_SEQUENCE) );
}


void MainWindow::updateRenderer()
{
    m_updateRenderer = true;
}


void MainWindow::updateImage()
{
    m_updateImage = true;
}

void MainWindow::loop()
{
    this->manager->getPipeline()->process(false);
}

MainWindow::~MainWindow()
{
    delete m_window;
    delete m_settings;
    delete manager;
}

void MainWindow::on_actionSettings_triggered()
{
    m_settings->show();
}

void MainWindow::on_btnProcess_clicked()
{
    renderer->trigger(EventManager::RUN);
}

void MainWindow::on_btnStep_clicked()
{
    renderer->trigger(EventManager::STEP);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    renderer->trigger(EventManager::CLOSE);
    process_thread->join();
    event->accept();
}

void MainWindow::on_btnPause_clicked()
{
    renderer->trigger(EventManager::PAUSE);
}

void MainWindow::on_btnReset_clicked()
{
    renderer->trigger(EventManager::RESET);
}

void MainWindow::on_cmbSequence_currentIndexChanged(int index)
{
    // TODO: fix this for more than two sequences

    if (index == 1)
    {
        renderer->trigger(EventManager::NEXT_SEQUENCE);
    }
    else
    {
        renderer->trigger(EventManager::PREVIOUS_SEQUENCE);
    }
}

void MainWindow::on_btnObjectPrevious_clicked()
{
    renderer->trigger(EventManager::PREVIOUS_OBJECT);
}

void MainWindow::on_btnObjectNext_clicked()
{
    renderer->trigger(EventManager::NEXT_OBJECT);
}

void MainWindow::on_btnSnapshot_clicked()
{
    renderer->trigger(EventManager::GRAB_FRAME);
}

void MainWindow::on_btnContinue_clicked()
{
    renderer->trigger(EventManager::CONTINUE);
}

void MainWindow::on_btnAddSequence_clicked()
{
    renderer->trigger(EventManager::ADD_SEQUENCE);
}

void MainWindow::on_btnSave_clicked()
{
    manager->applyParametersToConfig(manager->getConfig());
    manager->getConfig()->save();
}

void MainWindow::on_btnStartRecording_clicked()
{
    renderer->trigger(EventManager::START_RECORDING);
}

void MainWindow::on_btnStopRecording_clicked()
{
    renderer->trigger(EventManager::END_RECORDING);
}
