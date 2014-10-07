
#include "cameraReader.h"

#include <boost/asio.hpp>

namespace object_modeller
{
namespace reader
{

class TurntableReader : public CameraReader
{
private:
    int rpm;
    int nrSteps;
    int stepWidth;

    boost::asio::io_service io;
    boost::asio::serial_port *port;
    std::string port_name;

public:
    TurntableReader(boost::shared_ptr<output::Renderer> renderer, std::string config_name="turntablereader") : CameraReader(renderer, config_name)
    {
        grabber = NULL;
        port = NULL;

        registerParameter("nrSteps", "Number of steps", &nrSteps, 10);
        registerParameter("stepWidth", "Step width", &stepWidth, 0);
        registerParameter("portname", "Port name", &port_name, std::string("/dev/ttyACM0"));
    }

    void step();

    virtual std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process();

    std::string getName()
    {
        return "Turntable Reader";
    }
};

}
}
