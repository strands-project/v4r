/***************************************************************************
 *   Copyright (C) 2009 by Markus Bader   *
 *   markus.bader@austrian-kangaroos.com   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
/**
 * @file pase_args.cpp
 * @author Markus Bader
 * @brief
 **/

#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

namespace V4R {
bool parse_args(int argc, char* argv[], std::string &configFile, boost::program_options::options_description &desc) {

  boost::program_options::variables_map vm;
  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  } catch (const std::exception& ex) {
    std::cout << desc << "\n";
    return 1;
  }
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("configfile")) {
    std::ifstream file(configFile.c_str(), std::ifstream::in);
    if (file.is_open()) {
      try {
        boost::program_options::store(boost::program_options::parse_config_file(file, desc), vm);
        boost::program_options::notify(vm);
      } catch (const std::exception& ex) {
        std::cout << "Error reading config file: " << ex.what() << std::endl;
        return 1;
      }
    } else {
      std::cout << "Error opening config file " << configFile << std::endl;
      return 1;
    }
  }
  return 0;
}

}

