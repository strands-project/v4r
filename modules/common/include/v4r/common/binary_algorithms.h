/*
 * Author: Thomas Faeulhammer
 * Date: August 2015
 *
 */

#ifndef V4R_BINARY_ALGORITHMS_H__
#define V4R_BINARY_ALGORITHMS_H__

#include <v4r/core/macros.h>
#include <v4r/common/miscellaneous.h>
#include <vector>
#include <assert.h>
#include <stdlib.h>

namespace v4r
{

//    class V4R_EXPORTS BOOLEAN
//    {
//    public:
        enum V4R_EXPORTS BINARY_OPERATOR{
            AND,
            AND_N, // this will negate the second argument
            OR,
            OR_N, // this will negate the second argument
            XOR
        };

        /**
         * @brief performs bit wise logical operations
         * @param[in] bit mask1
         * @param[in] bit mask2
         * @param[in] operation (AND, AND_N, OR, XOR)
         * @return output bit mask
         */
        inline V4R_EXPORTS boost::dynamic_bitset<>
        binary_operation(const boost::dynamic_bitset<> &mask1, const boost::dynamic_bitset<> &mask2, int operation)
        {
            assert(mask1.size() == mask2.size());

            boost::dynamic_bitset<> output_mask ( mask1.size() );

            for(size_t i=0; i<mask1.size(); i++)
            {
                if(operation == BINARY_OPERATOR::AND)
                {
                    output_mask[i] = mask1[i] && mask2[i];
                }
                else
                if (operation == BINARY_OPERATOR::AND_N)
                {
                    output_mask[i] = mask1[i] && !mask2[i];
                }
                else
                if (operation == BINARY_OPERATOR::OR)
                {
                    output_mask[i] = mask1[i] || mask2[i];
                }
                else
                if (operation == BINARY_OPERATOR::OR_N)
                {
                    output_mask[i] = mask1[i] || !mask2[i];
                }
                else
                if (operation == BINARY_OPERATOR::XOR)
                {
                    output_mask[i] = (mask1[i] && !mask2[i]) || (!mask1[i] && mask2[i]);
                }
            }
            return output_mask;
        }


        /**
         * @brief given vector of indices of an image or pointcloud,
         * this function creates a boolean mask of the concatenated indices
         * @param vector of objct objectIndices
         * @param image_size
         * @return object bit mask
         */
        template<typename IdxT>
        inline
        boost::dynamic_bitset<>
        createMaskFromVecIndices( const typename std::vector< std::vector<IdxT> > &v_indices,
                                       size_t image_size)
        {
            boost::dynamic_bitset<> mask;

            if ( mask.size() != image_size )
                mask.resize( image_size, 0 );

            for(size_t i=0; i<v_indices.size(); i++)
            {
                boost::dynamic_bitset<> mask_tmp = v4r::createMaskFromIndices(v_indices[i], image_size);

                if( !mask.empty() )
                    mask = binary_operation(mask, mask_tmp, BINARY_OPERATOR::OR);
                else
                    mask = mask_tmp;
            }

            return mask;
        }

//    };
}
#endif
