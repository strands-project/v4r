#include <v4r/io/filesystem.h>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
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
createDirIfNotExist(const std::string & dirs)
{
    boost::filesystem::path dir = dirs;
    if(!boost::filesystem::exists(dir))
    {
        boost::filesystem::create_directories(dir);
    }
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
    const boost::filesystem::path source(src);
    const boost::filesystem::path destination(dst);

    namespace fs = boost::filesystem;

    try
    {
        // Check whether the function call is valid
        if( !fs::exists(source) || !fs::is_directory(source) )
        {
            std::cerr << "Source directory " << source.string() << " does not exist or is not a directory." << std::endl;
            return false;
        }

        if( fs::exists(destination) )
        {
            std::cerr << "Destination directory " << destination.string() << " already exists." << std::endl;
            return false;
        }

        // Create the destination directory
        if( !fs::create_directory(destination) )
        {
            std::cerr << "Unable to create destination directory"<< destination.string() << std::endl;
            return false;
        }
    }
    catch(fs::filesystem_error const & e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    // Iterate through the source directory
    for( fs::directory_iterator file(source); file != fs::directory_iterator(); ++file )
    {
        try
        {
            fs::path current(file->path());
            if(fs::is_directory(current))
            {
                // Found directory: Recursion
                if( !copyDir( current.string(), dst + "/" + current.filename().string() ) )
                    return false;
            }
            else // Found file: Copy
                fs::copy_file( current.string(), dst + "/" + current.filename().string() );
        }
        catch(fs::filesystem_error const & e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
    return true;
}

}

}

