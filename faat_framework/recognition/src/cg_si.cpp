

#include <faat_pcl/recognition/cg_si.h>


std::vector< std::vector <FeatureCorr> > GeometricConsistencySI (std::vector<cv::Point3f> refFeat, std::vector<cv::Point3f> trgFeat, std::vector<FeatureCorr> corresps, double consistency, int minConsensusSize)
{

	std::sort(corresps.begin(), corresps.end(), correspSorter);

	std::vector< std::vector <FeatureCorr> > outCorr;
	std::vector<int> consensusSet;
	bool *takenCorresps = (bool *)calloc( corresps.size(), sizeof(bool) );

	double distR[3], distT[3];

	for (int i=0; i<(int)corresps.size(); i++)
	{
		if ( takenCorresps[i])
			continue;
		
		for (int j=0; j<(int)corresps.size(); j++)
		{
			if ( j == i || takenCorresps[j])
				continue;
			
			consensusSet.clear();
			consensusSet.push_back(i);
			consensusSet.push_back(j);

			//assumption: scale is computed as the ratio of the target over the reference
			distR[0] = refFeat[corresps[i].refIndex].x - refFeat[corresps[j].refIndex].x;
			distR[1] = refFeat[corresps[i].refIndex].y - refFeat[corresps[j].refIndex].y;
			distR[2] = refFeat[corresps[i].refIndex].z - refFeat[corresps[j].refIndex].z;
			
			distT[0] = trgFeat[corresps[i].trgIndex].x - trgFeat[corresps[j].trgIndex].x;
			distT[1] = trgFeat[corresps[i].trgIndex].y - trgFeat[corresps[j].trgIndex].y;
			distT[2] = trgFeat[corresps[i].trgIndex].z - trgFeat[corresps[j].trgIndex].z;
			double scale = sqrt( (distR[0]*distR[0] + distR[1]*distR[1] + distR[2]*distR[2]) / (distT[0]*distT[0] + distT[1]*distT[1] + distT[2]*distT[2]) );
			
			for (int k=0; k<(int)corresps.size(); k++)
			{

				if ( k==i || k==j || takenCorresps[k])
					continue;
			
				//Let's check if k fits into the current consensus set
				bool isAGoodCandidate = true;
				for ( int c=0; c<(int)consensusSet.size(); c++)
				{
			
					distR[0] = refFeat[corresps[consensusSet[c]].refIndex].x - refFeat[corresps[k].refIndex].x;
					distR[1] = refFeat[corresps[consensusSet[c]].refIndex].y - refFeat[corresps[k].refIndex].y;
					distR[2] = refFeat[corresps[consensusSet[c]].refIndex].z - refFeat[corresps[k].refIndex].z;
			
					distT[0] = trgFeat[corresps[consensusSet[c]].trgIndex].x - trgFeat[corresps[k].trgIndex].x;
					distT[1] = trgFeat[corresps[consensusSet[c]].trgIndex].y - trgFeat[corresps[k].trgIndex].y;
					distT[2] = trgFeat[corresps[consensusSet[c]].trgIndex].z - trgFeat[corresps[k].trgIndex].z;

					double distance = abs( sqrt(distR[0]*distR[0] + distR[1]*distR[1] + distR[2]*distR[2]) - scale * sqrt(distT[0]*distT[0] + distT[1]*distT[1] + distT[2]*distT[2]) );

					if ( distance > consistency)
					{
						isAGoodCandidate = false;
						break;
					}	
				}
				if ( isAGoodCandidate ) 
					consensusSet.push_back( j );
			} //k loop

			if ((int)consensusSet.size() > minConsensusSize)
			{
				std::vector<FeatureCorr> tempOutCorr;
			
				for ( int k=0; k<(int)consensusSet.size(); k++)
				{
					tempOutCorr.push_back( corresps[ consensusSet[k] ] );
					takenCorresps[ consensusSet[k] ] = true;
				}
				outCorr.push_back( tempOutCorr );		
				break; //skip the current j-th loop and the current i-th element as the current i-th element has been already taken
			}
		} //j loop
	}//i loop
		
	free(takenCorresps);

	return outCorr;
}
