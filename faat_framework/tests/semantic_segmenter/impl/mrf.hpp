
namespace hombreViejo
{

struct MrfNode
{
	int index;
	//int label;
	float distance;
	//float score;
};


bool operator<(const MrfNode& a, const MrfNode& b)
{
	return a.distance < b.distance;
}

template <typename PointInT>
std::vector<std::vector<MrfNode> > createGraphKNN(const typename pcl::PointCloud<PointInT>::ConstPtr& feats, int k)
{
	std::vector<std::vector<MrfNode> > graph;
	graph.resize(feats->size());

	pcl::KdTreeFLANN<PointInT> kdtree;
	kdtree.setSortedResults(true);
	kdtree.setInputCloud( feats );
	std::vector<int> kdtree_indices;
	std::vector<float> kdtree_dists;

	for(unsigned i=0; i<feats->size(); i++)
	{
		kdtree_indices.clear();
		kdtree_dists.clear();
		kdtree.nearestKSearch(feats->at(i), k+1, kdtree_indices, kdtree_dists);

		graph[i].resize(k);
		for(int j=0; j<k; j++){

			graph[i][j].index = kdtree_indices[j+1];
			graph[i][j].distance = kdtree_dists[j+1];
			//discard first element cos its distance is 0 (same element)
			/*tempNode.index = neighbors[j+1].index;
			tempNode.distance = neighbors[j+1].distance;
			graph[i].push_back(tempNode);*/
		}
	}

	return graph;

}



template <typename PointInT>
std::vector<std::vector<MrfNode> > createGraphEBall(const typename pcl::PointCloud<PointInT>::ConstPtr& feats, double radius)
{

	std::vector<std::vector<MrfNode> > graph;
	graph.resize(feats->size());

	pcl::KdTreeFLANN<PointInT> kdtree;
	kdtree.setSortedResults(true);
	kdtree.setInputCloud( feats );
	std::vector<int> kdtree_indices;
	std::vector<float> kdtree_dists;

	for(unsigned i=0; i<feats->size(); i++)
	{
		kdtree_indices.clear();
		kdtree_dists.clear();
		kdtree.radiusSearch(feats->at(i), radius, kdtree_indices, kdtree_dists);

		graph[i].resize(kdtree_indices.size()-1);
		for(unsigned j=0; j<kdtree_indices.size()-1; j++){

			graph[i][j].index = kdtree_indices[j+1];
			graph[i][j].distance = kdtree_dists[j+1];
			//discard first element cos its distance is 0 (same element)
			/*tempNode.index = neighbors[j+1].index;
			tempNode.distance = neighbors[j+1].distance;

			graph[i].push_back(tempNode);*/
		}
	}

	return graph;
}



std::vector<std::vector<int> > computeConnComps(std::vector<std::vector<MrfNode> > & graph, pcl::PointCloud<pcl::PointXYZL>::Ptr labels, std::vector<std::vector<MrfNode> > & cutGraph)
{

	unsigned nFeat = graph.size();
	std::vector<std::vector<int> > connectedComponents;

	std::vector<std::vector<int> > tempConnectedComponents(nFeat);
	cutGraph.resize(nFeat);

	int index;
	int numConn = 0;
	std::queue<int> toExamin;

	int *examined = new int[nFeat];
	for (unsigned i=0; i<nFeat; i++)
		examined[i]=0;

	bool finish = false;

	for (unsigned i=0; i<nFeat; i++)
	{
		for (unsigned int j=0; j<graph[i].size(); j++)
		{
			if (labels->at(i).label == labels->at(graph[i][j].index).label )
			{
				cutGraph[i].push_back(graph[i][j]);
			}
		}
	}

	while(!finish)
	{
		for (unsigned i=0; i<nFeat; i++)
		{
			if(examined[i]==0)
			{
				toExamin.push(i);
				examined[i]=1;
				tempConnectedComponents[numConn].push_back(i);
				finish=false;
				break;
			}
			else
				finish=true;
		}

		if(!finish)
		{
			while(!toExamin.empty())
			{
				index = toExamin.front();
				toExamin.pop();

				for(unsigned int i=0; i<cutGraph[index].size(); i++)
					if(examined[cutGraph[index][i].index]==0)
					{
						examined[cutGraph[index][i].index] = 1;
						toExamin.push(cutGraph[index][i].index);
						tempConnectedComponents[numConn].push_back(cutGraph[index][i].index);
					}
			}
			numConn++;
		}

	}

	for(int i=0; i<numConn; i++)
		connectedComponents.push_back(tempConnectedComponents[i]);

	delete [] examined;

	return connectedComponents;
}


#define BOOST_THREAD_OPT

void belief_propagation_compute_new_messageSet(unsigned startIndex, unsigned finishIndex, float ***msg, float ***msg_old,
                                                std::vector<std::vector<MrfNode> > &graph, std::vector<std::vector<float> > &probs,
                                                float sigma)
{
	//COMPUTE NEW MESSAGE SET
	float temp;
	unsigned nClass = probs[0].size();
	float sq_sigma = sigma*sigma;

	for(unsigned q=startIndex; q<finishIndex; q++)
	{
		//for(p=0; p<nNeigh[q]; p++){
		for(unsigned p=0; p<graph[q].size(); p++)
		{

			//Computing msg[q][p] set - message passed from p to q at time t
			int neighbor_index = graph[q][p].index;
			double neighbor_distance = graph[q][p].distance;

			for(unsigned c1=0; c1<nClass; c1++)
			{

				msg[q][p][c1]=std::numeric_limits<float>::max();

				for(unsigned c2=0; c2<nClass; c2++)
				{
					//double compatibility = (c1 == c2) ? 0 : 1.0;
					float compatibility = (c1 == c2) ? 0 : exp(-neighbor_distance / sq_sigma);

					float sum_msgs = 0;
					for(unsigned n=0; n<graph[neighbor_index].size(); n++)
					{
						if( graph[neighbor_index][n].index != (int)q)
							sum_msgs += msg_old[neighbor_index][n][c2];
					}

					temp = compatibility + probs[neighbor_index][c2] + sum_msgs;

					if(temp < msg[q][p][c1])
						msg[q][p][c1] = temp;

				}//c2
			}//c1
		}//p

		//NORMALIZATION ?
		//for(p=0; p<nNeigh[q]; p++){
		for(unsigned p=0; p<graph[q].size(); p++)
		{
			float norm = 0.0;
			for(unsigned c1=0; c1<nClass; c1++)
				norm += msg[q][p][c1];
			norm = norm / nClass;
			for(unsigned c1=0; c1<nClass; c1++)
				msg[q][p][c1] -= norm;

		}//p for normalization
	}//q
}

//IMPROVEMENTS:
//The distance field could be filled-in with the geodesic distance in spite of the euclidean distance
void beliefPropagation(std::vector<std::vector<MrfNode> > & graph, std::vector<std::vector<float> > & probs, pcl::PointCloud<pcl::PointXYZL>::Ptr outCloud, float lambda, int iterations, float sigma, bool verbose){

	if ( probs.size() != graph.size() )
		std::cout << "Corrupt input data.." << probs.size() << " != " << graph.size() << std::endl;

	unsigned nFeat = graph.size();
	unsigned nClass = probs[0].size();

	float ***msg = new float **[nFeat];
	float ***msg_old = new float **[nFeat];

	for(unsigned q = 0; q<nFeat; q++)
	{
		msg[q] = new float *[graph[q].size()];
		msg_old[q] = new float *[graph[q].size()];

		for(unsigned p=0; p<graph[q].size(); p++)
		{
			msg[q][p] = new float[nClass];
			msg_old[q][p] = new float[nClass];

			memset(msg[q][p], 0, nClass * sizeof(float));
			memset(msg_old[q][p], 0, nClass * sizeof(float));
		}
	}

	float sq_sigma = sigma * sigma;

	////Fill in evidences
	for(unsigned q=0; q<nFeat; q++)
	{
		for(unsigned c=0; c<nClass; c++)
		{
			probs[q][c] = (1.0f-probs[q][c])*lambda;// * ((classes[q].label == p) ? (1.0-classes[q].score) : classes[q].score );
		}
	}

	for(int t=0; t<iterations; t++){

		//COMPUTE NEW MESSAGE SET
#ifdef BOOST_THREAD_OPT
		unsigned nThreads =  boost::thread::hardware_concurrency();
		unsigned step = nFeat / nThreads;

		std::vector<boost::thread*> threads;
		for (unsigned i = 0; i < nThreads-1; i++){
			threads.push_back( new boost::thread(belief_propagation_compute_new_messageSet,i*step,(i+1)*step, msg, msg_old, graph, probs, sq_sigma) );
		}
		threads.push_back( new boost::thread(belief_propagation_compute_new_messageSet,(nThreads-1)*step,nFeat, msg, msg_old, graph, probs, sq_sigma) );

		for (unsigned i = 0; i < threads.size(); i++)
		{
			threads[i]->join();
			delete threads[i];
		}
#else
		for(unsigned q=0; q<nFeat; q++)
		{
			for(unsigned p=0; p<graph[q].size(); p++)
			{

				//Computing msg[q][p] set - message passed from p to q at time t
				int neighbor_index = graph[q][p].index;
				double neighbor_distance = graph[q][p].distance;

				for(unsigned c1=0; c1<nClass; c1++)
				{
					//msg[q][p][c1] = min (c2) {comp(c2 to p, c1 to q) + evidence(c2 to p) + sum msgs (t-1) [c2]}
					msg[q][p][c1] = std::numeric_limits<float>::max();

					for(unsigned c2=0; c2<nClass; c2++)
					{

						//double compatibility = (c1 == c2) ? 0 : 1.0;
						double compatibility = (c1 == c2) ? 0 : exp(-neighbor_distance / sq_sigma);

						double sum_msgs = 0;
						for( unsigned n=0; n<graph[neighbor_index].size(); n++)
						{
							if( graph[neighbor_index][n].index != q)
								sum_msgs += msg_old[neighbor_index][n][c2];
						}

						float temp = compatibility + probs[neighbor_index][c2] + sum_msgs;

						if(temp < msg[q][p][c1])
							msg[q][p][c1] = temp;

					}//c2
				}//c1
			}//p

			//NORMALIZATION ?
			for(unsigned p=0; p<graph[q].size(); p++)
			{
				float norm = 0.0;
				for(unsigned c=0; c<nClass; c++)
					norm += msg[q][p][c];

				norm = norm / nClass;
				for(unsigned c=0; c<nClass; c++)
					msg[q][p][c] -= norm;

			}//p for normalization
		}//q
#endif
		//Message copy via pointer swapping
		float ***msg_temp = msg;
		msg = msg_old;
		msg_old = msg_temp;

		//Do this only in "verbose" mode - computing global energy for each iteration
		if(verbose)
		{
			for(unsigned q=0; q<nFeat; q++)
			{
				float belief_best = std::numeric_limits<float>::max();
				unsigned c_best = -1;

				for(unsigned c=0; c<nClass; c++)
				{
					float belief = probs[q][c];

					for(unsigned p=0; p<graph[q].size(); p++)
						belief += msg[q][p][c];

					if(belief < belief_best)
					{
						belief_best = belief;
						c_best = c;
					}
				}//c
				outCloud->at(q).label = c_best;
			}//q

			//global energy computation
			float energy = 0.0;
			for(unsigned q=0; q<nFeat; q++)
			{
				unsigned c_best = (unsigned)outCloud->at(q).label;
				energy += probs[q][c_best];

				//for(p=0; p<nNeigh[q]; p++)
				for(unsigned p=0; p<graph[q].size(); p++)
					energy += std::abs(int(c_best - outCloud->at( graph[q][p].index ).label));
			}

			std::cout << "Iter. "<< t << ": Free energy: " << energy << std::endl;
		} //Verbose mode

	} //t: end of iterations

	//Final best-label computation for each node
	for(unsigned q=0; q<nFeat; q++)
	{

		float belief_best = std::numeric_limits<float>::max();
		int c_best = -1;

		for(unsigned c=0; c<nClass; c++)
		{
			float belief = probs[q][c];

			for(unsigned p=0; p<graph[q].size(); p++)
				belief += msg[q][p][c];

			//update final probs as the last beliefs
			probs[q][c] = belief;

			if(belief < belief_best)
			{
				belief_best = belief;
				c_best = c;
			}
		}//c
		outCloud->at(q).label = c_best;

		//normalize out probs
		float totProbs = 0.0f;
		for(unsigned c=0; c<nClass; c++)
		{
			totProbs += probs[q][c];
		}
		for(unsigned c=0; c<nClass; c++)
		{
			probs[q][c] /= totProbs;
		}

	}//q

	//Do this only in "verbose" mode - computing final global energy
	if(verbose)
	{
		float energy = 0.0;
		for(unsigned q=0; q<nFeat; q++)
		{
			unsigned c_best = (unsigned)outCloud->at(q).label;
			energy += probs[q][c_best];

			//for(p=0; p<nNeigh[q]; p++)
			for(unsigned p=0; p<graph[q].size(); p++)
				energy += std::abs(int(c_best - outCloud->at( graph[q][p].index ).label));
		}

		std::cout << "Final free energy: " << energy << std::endl;
	} //verbose

	for(unsigned q=0; q<nFeat; q++)
	{
		for(unsigned p=0; p<graph[q].size(); p++)
		{
			delete [] msg[q][p];
			delete [] msg_old[q][p];
		}
		delete [] msg[q];
		delete [] msg_old[q];
	}
	delete [] msg;
	delete [] msg_old;

}

} // namespace hombreViejo

template<typename PointInT>
pcl::PointCloud<pcl::PointXYZL>::Ptr hombreViejo::solveMrfViaBP_kNN(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints,
                                                                std::vector<std::vector<float> > &probs,
                                                                std::vector<std::vector<int> > &ccomps,
                                                                float lambda, int iterations,
                                                                unsigned k_kNNGraph, bool verbose)
{
  float sigma_bp = 20.0f;

	std::cout << "Building graph (k-NN = " << k_kNNGraph << ") for MRF..." << std::flush;
	std::vector<std::vector<MrfNode> > graph = createGraphKNN<PointInT>(keypoints, k_kNNGraph);
	std::cout << "done" << std::endl;

	pcl::PointCloud<pcl::PointXYZL>::Ptr labelledCloud(new pcl::PointCloud<pcl::PointXYZL>);
	pcl::copyPointCloud(*keypoints, *labelledCloud);

	std::cout << "Launching MRF with lambda=" << lambda << ", nr.iter=" << iterations << ".." << std::flush;
	beliefPropagation(graph, probs, labelledCloud, lambda, iterations, sigma_bp, verbose);
	std::cout << "done" << std::endl;

	//compute connected components
	std::vector<std::vector<MrfNode> > cutGraph; //graph with no edges among different components; computed as a by-product of the computeConnComps function
	ccomps = computeConnComps(graph, labelledCloud, cutGraph);

  return labelledCloud;
}

template<typename PointInT>
pcl::PointCloud<pcl::PointXYZL>::Ptr hombreViejo::solveMrfViaBP_eBall(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints,
                                                                  std::vector<std::vector<float> > &probs,
                                                                  std::vector<std::vector<int> > &ccomps,
                                                                  float lambda, int iterations,
                                                                  double r_eBallGraph, bool verbose)
{
	float sigma_bp = 20.0f;

	std::cout << "Building graph (eBall = " << r_eBallGraph << ") for MRF..." << std::flush;
	std::vector<std::vector<MrfNode> > graph = createGraphEBall<PointInT>(keypoints, r_eBallGraph);
	std::cout << "done" << std::endl;

	pcl::PointCloud<pcl::PointXYZL>::Ptr labelledCloud(new pcl::PointCloud<pcl::PointXYZL>);
	pcl::copyPointCloud(*keypoints, *labelledCloud);

	std::cout << "Launching MRF with lambda=" << lambda << ", nr.iter=" << iterations << ".." << std::flush;
	beliefPropagation(graph, probs, labelledCloud, lambda, iterations, sigma_bp, verbose);
	std::cout << "done" << std::endl;

	//compute connected components
	std::vector<std::vector<MrfNode> > cutGraph; //graph with no edges among different components; computed as a by-product of the computeConnComps function
	ccomps = computeConnComps(graph, labelledCloud, cutGraph);

  return labelledCloud;
}

template<typename PointInT>
//pcl::PointCloud<pcl::PointXYZRGB>::Ptr hombreViejo::solveMrfViaBP(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints, std::vector<std::vector<float> > &probs, float lambda, int iterations, bool verbose)
pcl::PointCloud<pcl::PointXYZL>::Ptr hombreViejo::solveMrfViaBP(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints, std::vector<std::vector<float> > &probs, float lambda, int iterations, bool verbose)
{
  std::vector<std::vector<int> > ccomps;
	pcl::PointCloud<pcl::PointXYZL>::Ptr labelledCloud = solveMrfViaBP_kNN<PointInT>(keypoints, probs, ccomps, lambda, iterations, 4, verbose);


	//generate random colors
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr labelledCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>);

	//std::vector<pcl::RGB> colorPalette;
	//pcl::RGB tempColor;
	//for ( unsigned i=0; i<probs[0].size(); i++)
	//{
	//	tempColor.r = static_cast<unsigned char> (rand () % 256);
	//	tempColor.g = static_cast<unsigned char> (rand () % 256);
	//	tempColor.b = static_cast<unsigned char> (rand () % 256);

	//	colorPalette.push_back(tempColor);
	//}

	//int countSegments = 0;
	//pcl::PointXYZRGB tempRGBPoint;
	//for ( unsigned i=0; i<labelledCloud->size(); i++)
	//{
	//	tempRGBPoint.x = labelledCloud->at(i).x;
	//	tempRGBPoint.y = labelledCloud->at(i).y;
	//	tempRGBPoint.z = labelledCloud->at(i).z;

	//	tempRGBPoint.r = colorPalette[ labelledCloud->at(i).label ].r;
	//	tempRGBPoint.g = colorPalette[ labelledCloud->at(i).label ].g;
	//	tempRGBPoint.b = colorPalette[ labelledCloud->at(i).label ].b;

	//	labelledCloudRGB->push_back(tempRGBPoint);

	//	if (labelledCloud->at(i).label == 0)
	//		countSegments++;
	//}

	//std::cout << "Remaining segments: " << countPoles << std::endl;

	//return labelledCloudRGB;
	return labelledCloud;
}
