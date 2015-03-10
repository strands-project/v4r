/*
    Copyright (c) <year>, <copyright holder>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY <copyright holder> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "distributions.h"
#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>


namespace V4R {
boost::minstd_rand gIntGenStd;
boost::mt19937 gIntGen19937;
boost::variate_generator<boost::mt19937, boost::normal_distribution<> > gNormal(gIntGen19937, boost::normal_distribution<>(0,1));
boost::uniform_01<boost::minstd_rand> gUniform(gIntGenStd);

double  Distributions::normalDist(double mean, double sigma) {
    return mean + sigma * gNormal();
}

void Distributions::normalDist(cv::Vec<double,3> &mean, cv::Vec<double,3> sigma) {
    mean[0] = normalDist(mean[0], sigma[0]);
    mean[1] = normalDist(mean[1], sigma[1]);
    mean[2] = normalDist(mean[2], sigma[2]);
}


double Distributions::uniformDist(double min, double max){
    double s = max - min;
    return min + s * gUniform();
}

void Distributions::uniformDist(double min, double max, cv::Vec<double,3> &des){
    des[0] = uniformDist(min, max);
    des[1] = uniformDist(min, max);
    des[2] = uniformDist(min, max);
}


}

