#include "filesystem_utils.h"
#include <iostream>

using namespace v4r;

void
utils::getFoldersInDirectory (const bf::path & dir,
                              const std::string & rel_path_so_far,
                              std::vector<std::string> & relative_paths)
{
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
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
}


void utils::getFilesInDirectory (const bf::path & dir,
                                 std::vector<std::string> & relative_paths,
                                 const std::string & rel_path_so_far,
                                 const std::string & regex_pattern,
                                 bool recursive)
{
    const boost::regex file_filter( regex_pattern );

    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
    {
        //check if its a directory, then get files in it
        if (bf::is_directory (*itr))
        {
#if BOOST_FILESYSTEM_VERSION == 3
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
            std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

            bf::path curr_path = itr->path ();
            if (recursive)
            {
                getFilesInDirectory (curr_path, relative_paths, so_far, regex_pattern, recursive);
            }
        }
        else
        {
            //check for correct file pattern (extension,..) and then add, otherwise ignore..

#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = (itr->path ().filename ()).string ();
#else
            std::string file = (itr->path ()).filename ();
#endif

            boost::smatch what;
            if( !boost::regex_match( file, what, file_filter ) ) continue;

#if BOOST_FILESYSTEM_VERSION == 3
            std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
            std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

            relative_paths.push_back (path);
        }
    }
}


bool utils::writeMatrixToFile (const std::string &file, const Eigen::Matrix4f & matrix)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            out << matrix (i, j);
            if (!(i == 3 && j == 3))
                out << " ";
        }
    }
    out.close ();

    return true;
}

bool utils::readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix)
{

    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector < std::string > strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    for (int i = 0; i < 16; i++)
    {
        matrix (i / 4, i % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
    }

    return true;
}

bool utils::readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix, int padding)
{

    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector < std::string > strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    for (int i = 0; i < 16; i++)
    {
        matrix (i / 4, i % 4) = static_cast<float> (atof (strs_2[padding+i].c_str ()));
    }

    return true;
}

bool
utils::writeCentroidToFile (const std::string &file, const Eigen::Vector3f & centroid)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    out << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
    out.close ();

    return true;
}

bool
utils::getCentroidFromFile (const std::string &file, Eigen::Vector3f & centroid)
{
    std::ifstream in;

    in.open (file.c_str (), std::ifstream::in);

    if (!in)
    {
        std::cout << "Cannot open file " << file.c_str () << ".\n";
        return false;
    }

    char linebuf[256];
    in.getline (linebuf, 256);
    std::string line (linebuf);
    std::vector < std::string > strs;
    boost::split (strs, line, boost::is_any_of (" "));
    centroid[0] = static_cast<float> (atof (strs[0].c_str ()));
    centroid[1] = static_cast<float> (atof (strs[1].c_str ()));
    centroid[2] = static_cast<float> (atof (strs[2].c_str ()));

    return true;
}

bool
utils::writeFloatToFile (const std::string &file, const float value)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    out << value;
    out.close ();

    return true;
}

bool
utils::readFloatFromFile (const std::string &file, float& value)
{

    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    if (!in)
    {
        std::cout << "Cannot open file " << file.c_str () << ".\n";
        return false;
    }

    char linebuf[1024];
    in.getline (linebuf, 1024);
    value = static_cast<float> (atof (linebuf));

    return true;
}
