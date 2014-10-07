#include <boost/asio.hpp>

#define PORT_NAME "/dev/tty"

int main()
{
	boost::asio::io_service io;
	boost::asio::serial_port port( io, PORT_NAME );
	
	port.set_option( boost::asio::serial_port_base::baud_rate( 9600 ) );
	port.set_option( boost::asio::serial_port_base::character_size(8) );
	port.set_option( boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one) );
	port.set_option( boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none) );
	port.set_option( boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::none) );

	unsigned char command[1] = {0};
	char result;

	// write rpm
	command[0] = 'r';
	boost::asio::write(port, boost::asio::buffer(command, 1));
	command[0] = '1';
	boost::asio::write(port, boost::asio::buffer(command, 1));
	boost::asio::read(port, boost::asio::buffer(&result,1));
	
	std::cout << "result for rpm " << result << std::endl;
	
	char input;
	std::cin >> input;
	
	while (input != 'q') {
		
		command[0] = 's';
		boost::asio::write(port, boost::asio::buffer(command, 1));
		boost::asio::read(port, boost::asio::buffer(&result,1));
		
		std::cout << "result for step " << result << std::endl;
	
		std::cin >> input;
	}

	return 0;
}