#include <v4r/io/filesystem.h>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/regex.hpp>

namespace bf = boost::filesystem;

namespace v4r
{
namespace io
{

int
getFoldersInDirectory (const std::string & dir,
                       const std::string & rel_path_so_far,
                       std::vector<std::string> & relative_paths)
{
    bf::path dir_bf = dir;
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir_bf); itr != end_itr; ++itr)
    {
        //check if its a directory, else ignore
        if (bf::is_directory (*itr))
        {
#if BOOST_FILESYSTEM_VERSION == 3
            std::string path = rel_path_so_far + (itr->path ().filename ()).string();
#else
            std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif
            relative_paths.push_back (path);
        }
    }
    std::sort(relative_paths.begin(), relative_paths.end());
    return relative_paths.size();
}

std::vector<std::string>
getFoldersInDirectory (const std::string & dir)
{
    std::vector<std::string> relative_paths;

    bf::path dir_bf = dir;
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir_bf); itr != end_itr; ++itr)
    {
        if (bf::is_directory (*itr))
            relative_paths.push_back (itr->path ().filename ().string());
    }
    std::sort(relative_paths.begin(), relative_paths.end());

    return relative_paths;
}

int
getFilesInDirectory (const std::string &dir,
                     std::vector<std::string> & relative_paths,
                     const std::string & rel_path_so_far,
                     const std::string & regex_pattern,
                     bool recursive)
{
    bf::path dir_bf = dir;
    if ( !bf::is_directory ( dir_bf ) ) {
        std::cerr << dir << " is not a directory!" << std::endl;
        return -1;
    }
    const boost::regex file_filter( regex_pattern );

    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir_bf); itr != end_itr; ++itr)
    {
        //check if its a directory, then get files in it
        if (bf::is_directory (*itr))
        {
#if BOOST_FILESYSTEM_VERSION == 3
#ifdef _WIN32
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "\\";
#else
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#endif

#else
#ifdef _WIN32
            std::string so_far = rel_path_so_far + (itr->path ().filename ()) + "\\";
#else
            std::string so_far = rel_path_so_far + (itr->path ().filename ()) + "/";
#endif
#endif

            if (recursive)
            {
                if(getFilesInDirectory (itr->path().string(), relative_paths, so_far, regex_pattern, recursive) == -1)
                    return -1;
            }
        }
        else
        {
            //check for correct file pattern (extension,..) and then add, otherwise ignore..

#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = (itr->path ().filename ()).string ();
            std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
            std::string file = (itr->path ()).filename ();
            std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif
            boost::smatch what;
            if( boost::regex_match( file, what, file_filter ) )
                relative_paths.push_back (path);
        }
    }
    std::sort(relative_paths.begin(), relative_paths.end());
    return relative_paths.size();
}


std::vector<std::string>
getFilesInDirectory (const std::string &dir,
                     const std::string &regex_pattern,
                     bool recursive)
{
    std::vector<std::string> relative_paths;

    if ( !v4r::io::existsFolder( dir ) ) {
        std::cerr << dir << " is not a directory!" << std::endl;
    }
    else
    {
        bf::path bf_dir  = dir;
        bf::directory_iterator end_itr;
        for (bf::directory_iterator itr ( bf_dir ); itr != end_itr; ++itr)
        {
            const std::string file = itr->path().filename().string ();

            //check if its a directory, then get files in it
            if (bf::is_directory (*itr))
            {
                if (recursive)
                {
                    std::vector<std::string> files_in_subfolder  = getFilesInDirectory ( dir + "/" + file, regex_pattern, recursive);
                    for (const auto &sub_file : files_in_subfolder)
                        relative_paths.push_back( file + "/" + sub_file );
                }
            }
            else
            {
                //check for correct file pattern (extension,..) and then add, otherwise ignore..
                boost::smatch what;
                const boost::regex file_filter( regex_pattern );
                if( boost::regex_match( file, what, file_filter ) )
                    relative_paths.push_back ( file);
            }
        }
        std::sort(relative_paths.begin(), relative_paths.end());
    }
    return relative_paths;
}

bool
existsFile ( const std::string &rFile )
{
    bf::path dir_path = rFile;
    if ( bf::exists ( dir_path ) && bf::is_regular_file(dir_path)) {
        return true;
    } else {
        return false;
    }
}

bool
existsFolder ( const std::string &rFolder )
{
    bf::path dir = rFolder;
    return bf::exists (dir);
}

void
createDirIfNotExist(const std::string & dir)
{
    if(!bf::exists(dir))
        bf::create_directories(dir);
}


void
createDirForFileIfNotExist(const std::string & filename)
{
    if (filename.length())
    {
        size_t sep = filename.find_last_of("\\/");
        if (sep != std::string::npos)
        {
            std::string path = filename.substr(0, sep);
            createDirIfNotExist(path);
        }
    }
}


bool
copyDir(const std::string &src, const std::string &dst)
{
    const bf::path source(src);
    const bf::path destination(dst);

    try
    {
        // Check whether the function call is valid
        if( !bf::exists(source) || !bf::is_directory(source) )
        {
            std::cerr << "Source directory " << source.string() << " does not exist or is not a directory." << std::endl;
            return false;
        }

        if( bf::exists(destination) )
        {
            std::cerr << "Destination directory " << destination.string() << " already exists." << std::endl;
            return false;
        }

        // Create the destination directory
        if( !bf::create_directory(destination) )
        {
            std::cerr << "Unable to create destination directory"<< destination.string() << std::endl;
            return false;
        }
    }
    catch(bf::filesystem_error const & e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    // Iterate through the source directory
    for( bf::directory_iterator file(source); file != bf::directory_iterator(); ++file )
    {
        try
        {
            bf::path current(file->path());
            if(bf::is_directory(current))
            {
                // Found directory: Recursion
                if( !copyDir( current.string(), dst + "/" + current.filename().string() ) )
                    return false;
            }
            else // Found file: Copy
                bf::copy_file( current.string(), dst + "/" + current.filename().string() );
        }
        catch(bf::filesystem_error const & e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
    return true;
}

}

}

