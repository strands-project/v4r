#ifndef V4R_IO_EIGEN_H_
#define V4R_IO_EIGEN_H_

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

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

namespace Eigen
{
    template<class Matrix>
    V4R_EXPORTS
    inline void write_binary(const std::string &filename, const Matrix& matrix)
    {
        std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }
    template<class Matrix>
    V4R_EXPORTS
    inline void read_binary(const std::string &filename, Matrix& matrix)
    {
        std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }
}

#endif /* V4R_IO_EIGEN_H_ */
