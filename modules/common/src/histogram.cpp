#include <v4r/common/histogram.h>
#include <omp.h>

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


void
computeHistogram (const Eigen::MatrixXf &data, Eigen::MatrixXi &histogram, size_t bins, float min, float max)
{
    float bin_size = (max-min) / bins;
    int num_dim = data.cols();
    histogram = Eigen::MatrixXi::Zero (bins, num_dim);

    for (int dim = 0; dim < num_dim; dim++)
    {
        omp_lock_t bin_lock[bins];
        for(size_t pos=0; pos<bins; pos++)
            omp_init_lock(&bin_lock[pos]);

    #pragma omp parallel for firstprivate(min, max, bins) schedule(dynamic)
        for (int j = 0; j<data.rows(); j++)
        {
            int pos = std::floor( (data(j,dim) - min) / bin_size);

            if(pos < 0)
                pos = 0;

            if(pos > (int)bins)
                pos = bins - 1;

            omp_set_lock(&bin_lock[pos]);
            histogram(pos,dim)++;
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
}
