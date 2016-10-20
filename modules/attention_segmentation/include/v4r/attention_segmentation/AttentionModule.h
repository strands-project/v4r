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

#include "v4r/attention_segmentation/AttentionModuleErrors.h"
#include "v4r/attention_segmentation/BaseMap.h"
#include "v4r/attention_segmentation/ColorMap.h"
#include "v4r/attention_segmentation/OrientationMap.h"
#include "v4r/attention_segmentation/FrintropSaliencyMap.h"
#include "v4r/attention_segmentation/IKNSaliencyMap.h"
#include "v4r/attention_segmentation/MapsCombination.h"
#include "v4r/attention_segmentation/pyramidBase.h"
#include "v4r/attention_segmentation/pyramidSimple.h"
#include "v4r/attention_segmentation/pyramidItti.h"
#include "v4r/attention_segmentation/pyramidFrintrop.h"
#include "v4r/attention_segmentation/RelativeSurfaceOrientationMap.h"
#include "v4r/attention_segmentation/SurfaceCurvatureMap.h"
#include "v4r/attention_segmentation/SurfaceHeightMap.h"
#include "v4r/attention_segmentation/Symmetry3DMap.h"
#include "v4r/attention_segmentation/SymmetryMap.h"
#include "v4r/attention_segmentation/LocationMap.h"
#include "v4r/attention_segmentation/WTA.h"
#include "v4r/attention_segmentation/TJ.h"
#include "v4r/attention_segmentation/MSR.h"
#include "v4r/attention_segmentation/HitRatio.h"
#include "v4r/attention_segmentation/AmHitRatio.h"

#endif //ATTENTION_MODULE_INCLUDE_HEADERS_HPP
