#ifndef V4R_IO_EIGEN_H_
#define V4R_IO_EIGEN_H_

#include <string>
#include <vector>

#include <Eigen/Dense>

#include <v4r/core/macros.h>

namespace v4r
{
      namespace io
      {

        V4R_EXPORTS bool
        writeMatrixToFile (const std::string &file, const Eigen::Matrix4f & matrix);

        DEPRECATED(V4R_EXPORTS bool
        readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix, int padding=0));

        V4R_EXPORTS Eigen::Matrix4f
        readMatrixFromFile (const std::string &file, int padding=0);

        template<typename T>
        V4R_EXPORTS
        bool
        writeVectorToFile (const std::string &file, const typename std::vector<T> & centroid);

        V4R_EXPORTS bool
        getCentroidFromFile (const std::string &file, Eigen::Vector3f & centroid);

        V4R_EXPORTS bool
        writeFloatToFile (const std::string &file, const float value);

        V4R_EXPORTS bool
        readFloatFromFile (const std::string &file, float& value);

        V4R_EXPORTS bool
        is_number(const std::string& s);
      }

}

#endif /* V4R_IO_EIGEN_H_ */
