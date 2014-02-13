/*
 * valid_indices.h
 *
 *  Created on: Nov 12, 2012
 *      Author: aitor
 */

#ifndef VALID_INDICES_H_
#define VALID_INDICES_H_

__host__ __device__
int
getRectangleFromCorner (thrust::device_ptr<int> vol_, int x, int y, int z, int sidex, int sidey, int sidez, int m_width, int m_height)
{
  return vol_[(z + sidez - 1) * m_width * m_height + (y + sidey - 1) * m_width + x + sidex - 1] -
          vol_[(z + sidez - 1) * m_width * m_height + (y + sidey - 1) * m_width + x - 1] -
          vol_[(z + sidez - 1) * m_width * m_height + (y - 1) * m_width + x + sidex - 1] +
          vol_[(z + sidez - 1) * m_width * m_height + (y - 1) * m_width + x - 1] -
          vol_[(z - 1) * m_width * m_height + (y + sidey - 1) * m_width + x + sidex - 1] +
          vol_[(z - 1) * m_width * m_height + (y + sidey - 1) * m_width + x - 1] +
          vol_[(z - 1) * m_width * m_height + (y - 1) * m_width + x + sidex - 1] -
          vol_[(z - 1) * m_width * m_height + (y - 1) * m_width + x - 1];
}

struct validIdx
{
  int m_width;
  int m_height;
  int m_depth;
  int min_size_w_;
  int max_size_w_;
  int step_x_;
  int step_y_;
  int step_z_;
  int step_sx_;
  int step_sy_;
  int step_sz_;

  int size_angle_;
  int size_x_;
  int size_y_;
  int size_sx_;
  int size_sy_;

  thrust::device_ptr<int> rivs_occupancy;

  __host__ __device__
  validIdx (thrust::device_ptr<int> _rivs_occupancy,
            int _w, int _h, int _d, int _min_size_w_, int _max_size_w_, int _step_x_, int _step_y_, int _step_z_, int _step_sx_, int _step_sy_,
            int _step_sz_) :
              rivs_occupancy (_rivs_occupancy), m_width (_w), m_height (_h), m_depth (_d), min_size_w_ (_min_size_w_), max_size_w_ (_max_size_w_), step_x_ (_step_x_), step_y_ (_step_y_),
              step_z_ (_step_z_), step_sx_ (_step_sx_), step_sy_ (_step_sy_), step_sz_ (_step_sz_)
  {
    size_angle_ = ((m_width - min_size_w_) / step_x_) * ((m_height - min_size_w_) / step_y_) * ((max_size_w_ - min_size_w_) / step_sx_)
        * ((max_size_w_ - min_size_w_) / step_sy_) * ((min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);

    size_x_ = ((m_height - min_size_w_) / step_y_) * ((max_size_w_ - min_size_w_) / step_sx_) * ((max_size_w_ - min_size_w_) / step_sy_)
        * ((min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);

    size_y_ = ((max_size_w_ - min_size_w_) / step_sx_) * ((max_size_w_ - min_size_w_) / step_sy_) * ((min (max_size_w_ - min_size_w_,
                                                                                                                m_depth - min_size_w_)) / step_sz_);

    size_sx_ = ((max_size_w_ - min_size_w_) / step_sy_) * ((min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);

    size_sy_ = ((min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);
  }

  __host__ __device__
  int getAngle(const int idx) {
    return idx / size_angle_;
  }

  __host__ __device__
  int getX(const int idx) {
    return ((idx % size_angle_) / size_x_) * step_x_ + 1;
  }

  __host__ __device__
  int getY(const int idx) {
    return ((idx % size_angle_ % size_x_) / size_y_) * step_y_ + 1;
  }

  __host__ __device__
  int getZ(const int idx) {
    return 3;
  }

  __host__ __device__
  int getSX(const int idx) {
    return ((idx % size_angle_ % size_x_ % size_y_) / size_sx_) * step_sx_ + min_size_w_;
  }

  __host__ __device__
  int getSY(const int idx) {
    return ((idx % size_angle_ % size_x_ % size_y_ % size_sx_) / size_sy_) * step_sy_ + min_size_w_;
  }

  __host__ __device__
  int getSZ(const int idx) {
    return ((idx % size_angle_ % size_x_ % size_y_ % size_sx_ % size_sy_)) * step_sz_ + min_size_w_;
  }

  __host__ __device__
  bool
  operator() (const int idx)
  {
    if ((getX(idx) + getSX(idx)) >= m_width)
      return false;

    if ((getY(idx) + getSY(idx)) >= m_height)
      return false;

    if ((getZ(idx) + getSZ(idx)) >= m_depth)
      return false;

    if((getSX(idx) * getSY(idx) * getSZ(idx)) > 10000)
      return false;

    int vol = m_width * m_height * m_depth;
    int occupancy_val = getRectangleFromCorner (rivs_occupancy + getAngle(idx) * vol,
                                                getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx),
                                                m_width, m_height);

    if (occupancy_val <= 30)
          return false;

    if ((static_cast<float> (getRectangleFromCorner (rivs_occupancy + getAngle (idx) * vol, getX (idx), getY (idx),
                                                     getZ (idx) + getSZ (idx) / 3 * 2, getSX (idx), getSY (idx), max(getSZ (idx) / 3,1), m_width, m_height))
        / static_cast<float> (occupancy_val)) < 0.01f)
    {
      return false;
    }
    return true;
  }
};

#endif /* VALID_INDICES_H_ */
