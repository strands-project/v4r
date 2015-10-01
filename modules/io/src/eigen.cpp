#include <v4r/io/eigen.h>
#include <vector>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

namespace v4r
{
namespace io
{

bool
is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}


bool
writeMatrixToFile (const std::string &file, const Eigen::Matrix4f & matrix)
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

bool
readMatrixFromFile(const std::string &file, Eigen::Matrix4f & matrix, int padding)
{
    // check if file exists
    boost::filesystem::path path = file;
    if ( ! (boost::filesystem::exists ( path ) && boost::filesystem::is_regular_file(path)) )
        throw std::runtime_error ("Given file path to read Matrix does not exist!");


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
writeCentroidToFile (const std::string &file, const Eigen::Vector3f & centroid)
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
getCentroidFromFile (const std::string &file, Eigen::Vector3f & centroid)
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
writeFloatToFile (const std::string &file, const float value)
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
readFloatFromFile (const std::string &file, float& value)
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

}

}

