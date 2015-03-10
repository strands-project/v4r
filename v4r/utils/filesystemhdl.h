/***************************************************************************
 *   Copyright (C) 2010 by Markus Bader                                    *
 *   markus.bader@tuwien.ac.at                                             *
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
 * @file filesystemhdl.hpp
 * @author Markus Bader
 * @brief
 **/

#ifndef V4RFILESYSTEM_H
#define V4RFILESYSTEM_H

#include <string>
#include <vector>
#include <sys/time.h>

namespace V4R {

class FS {
public:
    /** Creates a folder
    * @param rFolder
    */
    static void createFolder ( const std::string &rFolder );
    
    /** Expands a file name including enviroment variables like $HOME
    * @param fileString
    * @return validated string
    */
    static std::string expandName ( const std::string &fileString );
    
    /** Returns a the name of files in a folder </br>
    * '(.*)bmp'
    * @param rFolder
    * @param rFiles
    * @param regExpressions examples "(.*)bmp",  "(.*)$"
    * @return Number of files
    */
    static int getFilesInFolder (const std::string &rFolder, std::vector<std::string> &rFiles, std::string regExpressions = "(.*)$" );

    /** checks if a file exists
    * @param rFile
    * @return true if file exsits
    */
    static bool existsFile ( const std::string &rFile );

    /** checks if a folder exists
    * @param rFolder
    * @return true if folder exsits
    */
    static bool existsFolder ( const std::string &rFolder );


    /** returns the filename which is the last part of rPathToFile
    * @param rPathToFile
    * @return filename
    */
    static std::string getFileNameOfPathToFile ( const std::string &rPathToFile );
    

    /** decodes the timestamp out of the filename like 2010-10-14--16-29-46--672.bmp
    * @param fileName
    * @param rTime
    * @return true on error
    */
    static bool timevalfromFileName (const std::string &fileName, timeval &rTime);
};
};

#endif
