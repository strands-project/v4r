/**
 *  Copyright (C) 2012  
 *    Ekaterina Potapova
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1040 Vienna, Austria
 *    potapova(at)acin.tuwien.ac.at
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */


#ifndef ATTENTION_MODULE_INCLUDE_HEADERS_HPP
#define ATTENTION_MODULE_INCLUDE_HEADERS_HPP

#include "v4r/attention_segmentation/AttentionModuleErrors.hpp"
#include "v4r/attention_segmentation/BaseMap.hpp"
#include "v4r/attention_segmentation/ColorMap.hpp"
#include "v4r/attention_segmentation/OrientationMap.hpp"
#include "v4r/attention_segmentation/FrintropSaliencyMap.hpp"
#include "v4r/attention_segmentation/IKNSaliencyMap.hpp"
#include "v4r/attention_segmentation/MapsCombination.hpp"
#include "v4r/attention_segmentation/pyramidBase.hpp"
#include "v4r/attention_segmentation/pyramidSimple.hpp"
#include "v4r/attention_segmentation/pyramidItti.hpp"
#include "v4r/attention_segmentation/pyramidFrintrop.hpp"
#include "v4r/attention_segmentation/RelativeSurfaceOrientationMap.hpp"
#include "v4r/attention_segmentation/SurfaceCurvatureMap.hpp"
#include "v4r/attention_segmentation/SurfaceHeightMap.hpp"
#include "v4r/attention_segmentation/Symmetry3DMap.hpp"
#include "v4r/attention_segmentation/SymmetryMap.hpp"
#include "v4r/attention_segmentation/LocationMap.hpp"
#include "v4r/attention_segmentation/WTA.hpp"
#include "v4r/attention_segmentation/TJ.hpp"
#include "v4r/attention_segmentation/MSR.hpp"
#include "v4r/attention_segmentation/HitRatio.hpp"
#include "v4r/attention_segmentation/AmHitRatio.hpp"

#endif //ATTENTION_MODULE_INCLUDE_HEADERS_HPP
