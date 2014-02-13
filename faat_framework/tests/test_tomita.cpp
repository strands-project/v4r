/*
 * test_tomita.cpp
 *
 *  Created on: Mar 6, 2013
 *      Author: aitor
 */

#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <faat_pcl/recognition/impl/cg/graph_geometric_consistency.hpp>

int
main (int argc, char ** argv)
{

  typedef boost::adjacency_matrix<boost::undirectedS, Vertex > Graph;
  //Graph correspondence_graph(9+5+7);
  Graph correspondence_graph(9);
  boost::add_edge (0, 1, correspondence_graph);
  boost::add_edge (1, 2, correspondence_graph);
  boost::add_edge (2, 3, correspondence_graph);
  boost::add_edge (3, 4, correspondence_graph);
  boost::add_edge (4, 5, correspondence_graph);
  boost::add_edge (3, 5, correspondence_graph);
  boost::add_edge (5, 6, correspondence_graph);
  boost::add_edge (5, 7, correspondence_graph);
  boost::add_edge (6, 7, correspondence_graph);
  boost::add_edge (6, 3, correspondence_graph);
  boost::add_edge (2, 7, correspondence_graph);
  boost::add_edge (3, 7, correspondence_graph);
  boost::add_edge (2, 8, correspondence_graph);
  boost::add_edge (1, 8, correspondence_graph);
  boost::add_edge (0, 8, correspondence_graph);

  /*for(int i=9; i < (9+7); i++)
  {
    for(int j=(i+1); j < (9+7); j++)
    {
      if(i != j)
      {
        boost::add_edge (i, j, correspondence_graph);
      }
    }
  }

  for(int i=16; i < (16+5); i++)
  {
    for(int j=(i+1); j < (16+5); j++)
    {
      if(i != j)
      {
        boost::add_edge (i, j, correspondence_graph);
      }
    }
  }*/

  Tomita<Graph> tom;
  tom.find_cliques(correspondence_graph);
  std::cout << "Number of cliques found by tomita..." << tom.getNumCliquesFound() << std::endl;

  typedef std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> VectorType;
  std::vector<VectorType> cliques_found;

  {
    pcl::ScopeTime t("cliques tomita");
    tom.getCliques(cliques_found);
  }

  for(size_t i=0; i < cliques_found.size(); i++)
  {
    for(size_t j=0; j < cliques_found[i].size(); j++)
    {
      std::cout << (cliques_found[i][j] + 1) << " ";
    }

    std::cout << std::endl;
  }

  size_t thres = static_cast<size_t> (3);
  size_t max_clique = 0;
  size_t n_cliques = 0;
  std::vector<std::vector< int> * > cliques;
  save_cliques<Graph> max_clique_vis (thres, max_clique, n_cliques, cliques);

  {
    pcl::ScopeTime t("cliques boost");
    boost::bron_kerbosch_all_cliques<Graph, save_cliques<Graph> > (correspondence_graph, max_clique_vis, 3);
  }

  for(size_t i=0; i < cliques.size(); i++)
  {
    for(size_t j=0; j < cliques[i]->size(); j++)
    {
      std::cout << (cliques[i]->at(j) + 1) << " ";
    }

    std::cout << std::endl;
  }
}
