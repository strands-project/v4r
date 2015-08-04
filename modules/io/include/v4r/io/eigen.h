#ifndef V4R_IO_EIGEN_H_
#define V4R_IO_EIGEN_H_

#include <string>

#include <Eigen/Dense>

#include <v4r/core/macros.h>

namespace v4r
{
      namespace io
      {

        V4R_EXPORTS bool
        writeMatrixToFile (const std::string &file, const Eigen::Matrix4f & matrix);

        V4R_EXPORTS bool
        readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix);

        V4R_EXPORTS bool
        readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix, int padding);

        V4R_EXPORTS bool
        writeCentroidToFile (const std::string &file, const Eigen::Vector3f & centroid);

        V4R_EXPORTS bool
        getCentroidFromFile (const std::string &file, Eigen::Vector3f & centroid);

        V4R_EXPORTS bool
        writeFloatToFile (const std::string &file, const float value);

        V4R_EXPORTS bool
        readFloatFromFile (const std::string &file, float& value);

      }

}

#endif /* V4R_IO_EIGEN_H_ */
