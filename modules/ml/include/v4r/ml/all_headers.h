/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#pragma once

#include <v4r/ml/types.h>

#include <v4r/ml/nearestNeighbor.h>
#include <v4r/ml/svmWrapper.h>

namespace v4r
{

Classifier::Ptr
initClassifier(int method, std::vector<std::string> &params )
{
    Classifier::Ptr classifier;

    if(method == ClassifierType::KNN )
    {
        NearestNeighborClassifierParameter param;
        params = param.init(params);
        NearestNeighborClassifier::Ptr nn (new NearestNeighborClassifier (param));
        classifier = boost::dynamic_pointer_cast<Classifier > (nn);
    }
    else if(method == ClassifierType::SVM)
    {
        SVMParameter param;
        params = param.init(params);
        svmClassifier::Ptr nn (new svmClassifier (param));
        classifier = boost::dynamic_pointer_cast<Classifier > (nn);
    }
    else
    {
        std::cerr << "Classifier method " << method << " is not implemented! " << std::endl;
    }

    return classifier;
}


}
