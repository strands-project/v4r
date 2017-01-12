#include <v4r/appearance/histogram_equalization.h>
#include <v4r/common/histogram.h>

namespace v4r
{
void
HistogramEqualizer::equalize(const Eigen::VectorXf &input, Eigen::VectorXf &output)
{
    Eigen::MatrixXi hist;
    computeHistogram(input, hist, 256, 0.f, 100.f);
}
}
