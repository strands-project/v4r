/*
 * GtLinksFile.h
 *
 *  Created on: 15.12.2015
 *      Author: ivelas
 */

#ifndef GTLINKSFILE_H_
#define GTLINKSFILE_H_

#include <iostream>
#include <vector>
#include <string>

#include <boost/algorithm/string.hpp>

class GtLinksFile {
public:
	GtLinksFile(const std::string &fn) {
		in_file.open(fn.c_str());
		if(!in_file.is_open()) {
			perror(fn.c_str());
			exit(1);
		}
	}

	std::vector<std::string> getSequence() {
		std::vector<std::string> annotation_sequence;
		std::string line;
		std::getline(in_file, line);
		boost::split(annotation_sequence, line, boost::is_any_of(" "));
		return annotation_sequence;
	}

private:
	std::ifstream in_file;
};


#endif /* GTLINKSFILE_H_ */
