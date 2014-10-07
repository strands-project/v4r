
#include "reader/turntableReader.h"

#include <pcl/io/pcd_io.h>

#include <boost/lexical_cast.hpp>

namespace object_modeller
{
namespace reader
{

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> TurntableReader::process()
{
    if (grabber == NULL)
    {
        grabber = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
                    boost::bind(&CameraReader::grabberCallback, this, _1);
        grabber->registerCallback(f);

        boost::function < void (const boost::shared_ptr<openni_wrapper::Image>&)> f2 =
                    boost::bind (&CameraReader::image_callback, this, _1);
        grabber->registerCallback(f2);
    }

    if (stepWidth < 1)
    {
        stepWidth = 360 / nrSteps;
    }

    if (port == NULL)
    {
        port = new boost::asio::serial_port(io, port_name);

        port->set_option( boost::asio::serial_port_base::baud_rate( 9600 ) );
        port->set_option( boost::asio::serial_port_base::character_size(8) );
        port->set_option( boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one) );
        port->set_option( boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none) );
        port->set_option( boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::none) );


        // write step width
        char command[1];
        char result;
        std::string widthString = boost::lexical_cast<std::string>(stepWidth);

        command[0] = 'w';
        boost::asio::write(*port, boost::asio::buffer(command, 1));
        boost::asio::write(*port, boost::asio::buffer(widthString.c_str(), widthString.length()));
        boost::asio::read(*port, boost::asio::buffer(&result,1));
    }

    grabber->start();

    for (int i=0;i<nrSteps;i++)
    {
        copyFrame = true;

        //while (copyFrame) {}

        // step
        char command[1];
        char result;
        command[0] = 's';
        boost::asio::write(*port, boost::asio::buffer(command, 1));
        boost::asio::read(*port, boost::asio::buffer(&result,1));

        std::cout << "snapshot" << std::endl;
    }

    grabber->stop();
}

}
}
