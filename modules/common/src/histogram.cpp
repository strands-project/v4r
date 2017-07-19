#include <glog/logging.h>
#include <omp.h>
#include <v4r/common/histogram.h>

namespace v4r
{

int
computeHistogramIntersection (const Eigen::VectorXi &histA, const Eigen::VectorXi &histB)
{
    Eigen::MatrixXi histAB (histA.rows(), 2);
    histAB.col(0) = histA;
    histAB.col(1) = histB;

    Eigen::VectorXi minv = histAB.rowwise().minCoeff();
    return minv.sum();
}

void computeCumulativeHistogram (const Eigen::VectorXi &histogram, Eigen::VectorXi &cumulative_histogram)
{
    cumulative_histogram = Eigen::VectorXi::Zero(histogram.rows());
    cumulative_histogram(0) = histogram(0);

    for(int i=1; i<histogram.rows(); i++)
        cumulative_histogram(i) = cumulative_histogram(i-1) + histogram(i);
}


void
computeHistogram (const Eigen::MatrixXf &data, Eigen::MatrixXi &histogram, size_t bins, float min, float max)
{
    float bin_size = (max-min) / bins;
    int num_dim = data.cols();
    histogram = Eigen::MatrixXi::Zero (num_dim, bins);

    for (int dim = 0; dim < num_dim; dim++)
    {
        omp_lock_t bin_lock[bins];
        for(size_t pos=0; pos<bins; pos++)
            omp_init_lock(&bin_lock[pos]);

    #pragma omp parallel for firstprivate(min, max, bins) schedule(dynamic)
        for (int j = 0; j<data.rows(); j++)
        {
            int pos = std::max<int>(0, std::min<int>( bins-1, std::floor( (data(j,dim) - min) / bin_size) ) );

            omp_set_lock(&bin_lock[pos]);
            histogram(dim, pos)++;
            omp_unset_lock(&bin_lock[pos]);
        }

        for(size_t pos=0; pos<bins; pos++)
            omp_destroy_lock(&bin_lock[pos]);
    }
}

void
shiftHistogram (const Eigen::VectorXi &hist, Eigen::VectorXi &hist_shifted, bool direction)
{
    int bins = hist.rows();
    hist_shifted = Eigen::VectorXi::Zero(bins);

    if(direction){ //shift right
        hist_shifted.tail(bins - 1) = hist.head(bins-1);
        hist_shifted(bins-1) +=  hist(bins-1);
    }
    else { // shift left
        hist_shifted.head(bins - 1) = hist.tail(bins-1);
        hist_shifted(0) +=  hist(0);
    }
}

Eigen::VectorXf
specifyHistogram (const Eigen::VectorXf &input_image, const Eigen::VectorXf &desired_image, size_t bins, float min, float max)
{
    CHECK(min<max && bins>0);

    Eigen::MatrixXi hx, hz; //model color histogram is input, scene color histogram is desired
    computeHistogram(input_image, hx, bins, min, max);
    computeHistogram(desired_image, hz, bins, min, max);

    Eigen::VectorXi Hxi, Hzi;
    computeCumulativeHistogram(hx.col(0), Hxi);
    computeCumulativeHistogram(hz.col(0), Hzi);

    Eigen::VectorXf Hx = Hxi.cast<float>(), Hz = Hzi.cast<float>();

    int num_x = Hx.tail(1)(0);
    int num_z = Hz.tail(1)(0);
    Hx /= num_x; // normalize
    Hz /= num_z; // normalize

    int j=0;
    std::vector<size_t> lut (bins, 0);
    for(size_t i=0; i<bins; i++)
    {
        if( Hx[i] <= Hz[j] )
            lut[i] = j;
        else
        {
            while( j+1 < Hz.rows() && Hx[i] > Hz[j])
                j++;

            if( Hz[j] - Hx[i] > Hx[i] - Hz[j-1] )
                lut[i] = j-1;
            else
                lut[i] = j;
        }
    }

    float bin_size = (max-min) / bins;

    Eigen::VectorXf color_new ( input_image.rows() );
    for(int i=0; i<input_image.rows(); i++)
    {
        float color = input_image(i);
        int pos = std::floor( (color - min) / bin_size );

        if(pos < 0)
            pos = 0;

        if(pos > (int)bins)
            pos = bins - 1;

        int desired_bin = lut[pos];
        float new_color = desired_bin * bin_size + min;
        color_new(i) = new_color;
    }
    return color_new;
}



}
