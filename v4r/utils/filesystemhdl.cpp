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
 * @file filesystemhdl.cpp
 * @author Markus Bader
 * @brief
 **/

#define BOOST_FILESYSTEM_VERSION 2

#include "filesystemhdl.h"
#include <iostream>
#include <algorithm>
#include <cctype>

#include <boost/regex.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/thread/thread.hpp>
#include <wordexp.h>

#define FSL "/"

using namespace std;


/** compares to strings
* @param A
* @param B
* @return return false on a match
*/
inline bool compareCaseInsensitive( const std::string &left, const std::string &right ) {
    for ( std::string::const_iterator lit = left.begin(), rit = right.begin(); lit != left.end() && rit != right.end(); ++lit, ++rit ) {
        if ( tolower( *lit ) < tolower( *rit ) ) {
            return true;
        } else if ( tolower( *lit ) > tolower( *rit ) ) {
            return false;
        }
    }
    if ( left.size() < right.size() ) {
        return true;
    }
    return false;
}

using namespace V4R;

std::string FS::expandName ( const std::string &fileString ) {
  std::stringstream ss;
    wordexp_t p;
    char** w;
    wordexp( fileString.c_str(), &p, 0 );
    w = p.we_wordv;
    for (size_t i=0; i < p.we_wordc; i++ ) {
      ss << w[i];
    }
    wordfree( &p );
    return ss.str();
}

std::string FS::getFileNameOfPathToFile ( const std::string &rPathToFile ) {
    int start = rPathToFile.rfind(FSL);
    if (start == -1) start = 0;
    else start += 1;
    std::string filename = rPathToFile.substr(start);
    return filename;
}

bool FS::existsFile ( const string &rFile ) {
    boost::filesystem::path dir_path = boost::filesystem::complete ( boost::filesystem::path ( rFile, boost::filesystem::native ) );
    if ( boost::filesystem::exists ( dir_path ) && boost::filesystem::is_regular_file(dir_path)) {
        return true;
    } else {
        return false;
    }
}


bool FS::existsFolder ( const string &rFolder ) {
    boost::filesystem::path dir_path = boost::filesystem::complete ( boost::filesystem::path ( rFolder, boost::filesystem::native ) );
    if ( boost::filesystem::exists ( dir_path ) && boost::filesystem::is_directory(dir_path)) {
        return true;
    } else {
        return false;
    }
}

void FS::createFolder ( const string &rFolder ) {
    boost::filesystem::path dir_path = boost::filesystem::complete ( boost::filesystem::path ( rFolder, boost::filesystem::native ) );
    if ( !boost::filesystem::exists ( dir_path ) || !boost::filesystem::is_directory ( dir_path ) ) {
        boost::filesystem::create_directory ( dir_path );
    }
}

int FS::getFilesInFolder ( const string &rFolder,  vector<string> &rFiles, const string regx) {
    using namespace boost::filesystem;
    path fullPath = system_complete ( path ( rFolder.c_str(), native ) );

    if ( !exists ( fullPath ) ) {
        cerr << "Error: the directory " << fullPath.string( ) << " does not exist.\n";
        return ( -1 );
    }
    if ( !is_directory ( fullPath ) ) {
        cout << fullPath.string( ) << " is not a directory!\n";
        return ( -1 );
    }

    boost::regex_constants::syntax_option_type flags = boost::regex_constants::perl;
    // boost::regex_constants::syntax_option_type flags = boost::regex_constants::extended;
    // boost::regex_constants::syntax_option_type flags = boost::regex_constants::basic;


    static const boost::regex expression( regx, flags );
    int nrOfFiles = 0;
    directory_iterator end;
    for ( directory_iterator it ( fullPath ); it != end; ++it ) {
        string filename = it->filename();
        if ( !is_directory ( *it ) && boost::regex_match ( filename, expression ) ) {
            string fileNameFull = it->string();
            rFiles.push_back ( fileNameFull );
            //cout << it->filename() << endl;
            nrOfFiles++;
        }
    }
    sort( rFiles.begin(), rFiles.end(), compareCaseInsensitive );
    return nrOfFiles;
}
#include <boost/date_time/posix_time/posix_time.hpp>
bool FS::timevalfromFileName (const std::string &fileName, timeval &rTime) {
    using namespace boost::posix_time;
    using namespace boost::gregorian;
    std::string str = getFileNameOfPathToFile(fileName);
    int YYYY=0, MM=0, DD=0, hh=0, mm=0, ss=0, mls=0;
    int readCount = sscanf ( str.c_str(),"%d-%d-%d--%d-%d-%d--%d",&YYYY, &MM, &DD, &hh, &mm, &ss, &mls );
    if (readCount != 7) return true;
    ptime t(date(YYYY,MM,DD),hours(hh)+minutes(mm)+seconds(ss)+milliseconds(mls));
    ptime timet_start(date(1970,1,1));
    time_duration diff = t - timet_start;
    rTime.tv_sec = diff.ticks()/time_duration::rep_type::res_adjust();
    rTime.tv_usec = diff.fractional_seconds();
    return false;
}

