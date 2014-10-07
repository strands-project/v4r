#pragma once

#include <string>
#include <iostream>

#include <boost/tuple/tuple.hpp>

#include "result.h"
#include "config.h"

namespace object_modeller
{

class SequenceInfo
{
public:
    int nr_sequences;
    std::vector<int> sequence_lengths;

    SequenceInfo()
    {
        nr_sequences = 1;
        /*
        nrSequences = config.getInt("pipeline", "nrSequences", 1);
        std::string sequenceLengthString = config.getString("pipeline", "sequenceLengths", "");

        std::vector<std::string> lengths;
        boost::algorithm::split(lengths, sequenceLengthString, boost::algorithm::is_any_of(";"));

        for (unsigned int i=0;i<lengths.size();i++)
        {
            boost::algorithm::trim(lengths[i]);

            sequenceLengths.push_back(lengths[i]);
        }
        */
    }
};

}
