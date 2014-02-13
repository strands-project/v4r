/*
 * ggo.h
 *
 *  Created on: Apr 17, 2013
 *      Author: aitor
 */

#ifndef GGO_H_
#define GGO_H_

#include <pcl/common/common.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include "pcl/recognition/3rdparty/metslib/mets.hh"
#include <boost/graph/copy.hpp>
#include <boost/graph/breadth_first_search.hpp>

namespace faat_pcl
{
  namespace object_modelling
  {

    class bfs_time_visitor : public boost::default_bfs_visitor
    {
      bool & found_;
      int & vertex_;
    public:
      bfs_time_visitor(bool found, int vertex) : found_(found), vertex_(vertex)
      {

      }

      template < typename Vertex, typename Graph >
        void discover_vertex(Vertex u, const Graph & g) const
      {
        if(static_cast<int>(u) == vertex_)
        {
          //std::cout << static_cast<int>(u) << " " << vertex_ << std::endl;
          found_ = true;
        }
      }

      bool getFound()
      {
        return found_;
      }
    };

    template <typename PointT>
    class ggo
    {
    private:

      typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
      typedef typename pcl::PointCloud<PointT> PointCloud;
      typedef typename pcl::PointCloud<pcl::Normal>::Ptr PointCloudNormalPtr;
      typedef boost::property<boost::edge_weight_t, std::pair<int, float> > EdgeWeightProperty;
      typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, int, EdgeWeightProperty> DirectedGraph;
      typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, int, EdgeWeightProperty> Graph;

      std::vector<PointCloudPtr> clouds_;
      std::vector<PointCloudPtr> processed_clouds_;
      std::vector<PointCloudPtr> range_images_;
      float focal_length_;
      float cx_;
      float cy_;

      Graph LR_graph_;
      Graph ST_graph_;

      struct pair_wise_registration
      {
        float reg_error_;
        int overlap_;
        float overlap_factor_;
        float fsv_fraction_;
        float osv_fraction_;
        float fsv_local_;
        float loop_closure_probability_;
        float color_error_;
      };

      std::vector<std::vector<std::vector<Eigen::Matrix4f> > > pairwise_poses_;
      std::vector<std::vector<std::vector<pair_wise_registration> > > pairwise_registration_;
      std::vector< Eigen::Matrix4f > absolute_poses_;
      std::vector<bool> cloud_registered_;

      void
      visualizePairWiseAlignment ();
      void
      computeFSVFraction ();

      void computeAbsolutePosesFromMST(Graph & mst,
                                           boost::graph_traits < Graph >::vertex_descriptor & root,
                                           std::vector<Eigen::Matrix4f> & absolutes_poses,
                                           std::vector<boost::graph_traits < Graph >::vertex_descriptor> & cc);

      void computeAbsolutePosesRecursive(Graph & mst,
                                             boost::graph_traits < Graph >::vertex_descriptor & start,
                                             boost::graph_traits < Graph >::vertex_descriptor & coming_from,
                                             Eigen::Matrix4f accum,
                                             std::vector<Eigen::Matrix4f> & absolutes_poses,
                                             std::vector<boost::graph_traits < Graph >::vertex_descriptor> & cc,
                                             std::vector<boost::graph_traits<Graph>::vertex_descriptor> & graph_to_cc);

      void
      computeHuberST(Graph & local_registration,
                       Graph & mst);


      void refineAbsolutePoses();

      void visualizeGlobalAlignmentCCAbsolutePoses(Graph & local_registration,
                                          Graph & mst);

      class SAModel : public mets::evaluable_solution
      {
      public:
        std::vector<bool> solution_;
        ggo * opt_;
        mets::gol_type cost_;
        Graph LR_graph_;
        Graph ST_;
        std::vector< boost::graph_traits<Graph>::edge_descriptor> edge_iterators_;

        //Evaluates the current solution
        mets::gol_type
        cost_function () const
        {
          return cost_;
        }

        void
        copy_from (const mets::copyable& o)
        {
          const SAModel& s = dynamic_cast<const SAModel&> (o);
          solution_ = s.solution_;
          opt_ = s.opt_;
          cost_ = s.cost_;
          LR_graph_.clear();
          ST_.clear();
          boost::copy_graph (s.LR_graph_, LR_graph_);
          boost::copy_graph (s.ST_, ST_);
          edge_iterators_ = s.edge_iterators_;
        }

        void
        copy_from (const mets::feasible_solution& o)
        {
          const SAModel& s = dynamic_cast<const SAModel&> (o);
          solution_ = s.solution_;
          opt_ = s.opt_;
          cost_ = s.cost_;
          LR_graph_.clear();
          ST_.clear();
          boost::copy_graph (s.LR_graph_, LR_graph_);
          boost::copy_graph (s.ST_, ST_);
          edge_iterators_ = s.edge_iterators_;
        }

        mets::gol_type
        what_if (int /*index*/, bool /*val*/) const
        {
          return static_cast<mets::gol_type> (0);
        }

        mets::gol_type
        apply_and_evaluate (int index, bool val)
        {
          solution_[index] = val;
          if(val)
          {
            //std::cout << "Adding edge to the graph..." << std::endl;
            boost::add_edge(static_cast<int>(boost::source(edge_iterators_[index], LR_graph_)), static_cast<int>(boost::target(edge_iterators_[index], LR_graph_)), ST_);
          }
          else
            boost::remove_edge(static_cast<int>(boost::source(edge_iterators_[index], LR_graph_)), static_cast<int>(boost::target(edge_iterators_[index], LR_graph_)), ST_);

          mets::gol_type sol = opt_->evaluateSolution (solution_, index, edge_iterators_, ST_); //this will update the state of the solution
          cost_ = sol;
          return sol;
        }

        void
        apply (int /*index*/, bool /*val*/)
        {

        }

        void
        unapply (int index, bool val)
        {
          solution_[index] = val;

          if(val)
            boost::add_edge(static_cast<int>(boost::source(edge_iterators_[index], LR_graph_)), static_cast<int>(boost::target(edge_iterators_[index], LR_graph_)), ST_);
          else
            boost::remove_edge(static_cast<int>(boost::source(edge_iterators_[index], LR_graph_)), static_cast<int>(boost::target(edge_iterators_[index], LR_graph_)), ST_);

          //update optimizer solution
          cost_ = opt_->evaluateSolution (solution_, index, edge_iterators_, ST_); //this will udpate the cost function in opt_
        }

        void
        setSolution (std::vector<bool> & sol)
        {
          solution_ = sol;
        }

        void
        setLRGraph(Graph & g)
        {
          LR_graph_.clear();
          ST_.clear();
          boost::copy_graph (g, LR_graph_);
          boost::graph_traits<Graph>::edge_iterator ei, ei_end;
          for (boost::tie (ei, ei_end) = boost::edges (LR_graph_); ei != ei_end; ++ei)
          {
            edge_iterators_.push_back(*ei);
          }

          Graph ST(num_vertices(LR_graph_));
          boost::copy_graph (ST, ST_);
        }

        void
        setOptimizer (ggo * opt)
        {
          opt_ = opt;
        }
      };

      class generic_move : public mets::mana_move
      {
      public:
        virtual mets::gol_type
        evaluate (const mets::feasible_solution& cs) const = 0;
        virtual mets::gol_type
        apply_and_evaluate (mets::feasible_solution& cs) = 0;
        virtual void
        apply (mets::feasible_solution& s) const = 0;
        virtual void
        unapply (mets::feasible_solution& s) const = 0;
        virtual mets::clonable*
        clone () const = 0;
        virtual size_t
        hash () const = 0;
        virtual bool
        operator== (const mets::mana_move&) const = 0;
      };

      /*
       * Represents a move, activate/deactivate an hypothesis
       */

      class move : public generic_move
      {
        int index_;
      public:
        move (int i) :
          index_ (i)
        {
        }

        int
        getIndex ()
        {
          return index_;
        }

        mets::gol_type
        evaluate (const mets::feasible_solution& cs) const
        {
          //mets::copyable copyable = dynamic_cast<mets::copyable> (&cs);
          SAModel model;
          model.copy_from (cs);
          mets::gol_type cost = model.apply_and_evaluate (index_, !model.solution_[index_]);
          model.apply_and_evaluate (index_, !model.solution_[index_]);
          return cost;
        }

        mets::gol_type
        apply_and_evaluate (mets::feasible_solution& cs)
        {
          SAModel& model = dynamic_cast<SAModel&> (cs);
          return model.apply_and_evaluate (index_, !model.solution_[index_]);
        }

        void
        apply (mets::feasible_solution& s) const
        {
          SAModel& model = dynamic_cast<SAModel&> (s);
          model.apply_and_evaluate (index_, !model.solution_[index_]);
        }

        void
        unapply (mets::feasible_solution& s) const
        {
          SAModel& model = dynamic_cast<SAModel&> (s);
          model.unapply (index_, !model.solution_[index_]);
        }

        mets::clonable*
        clone () const
        {
          move * m = new move (index_);
          return static_cast<mets::clonable*> (m);
        }

        size_t
        hash () const
        {
          return static_cast<size_t> (index_);
        }

        bool
        operator== (const mets::mana_move& m) const
        {
          const move& mm = dynamic_cast<const move&> (m);
          return mm.index_ == index_;
        }
      };

      class move_manager
      {
      public:
        std::vector<generic_move*> moves_m;
        typedef typename std::vector<generic_move*>::iterator iterator;

        iterator
        begin ()
        {
          return moves_m.begin ();
        }
        iterator
        end ()
        {
          return moves_m.end ();
        }

        move_manager (Graph & G)
        {
        }

        ~move_manager ()
        {

        }

        void
        refresh (mets::feasible_solution& s)
        {
          SAModel& model = dynamic_cast<SAModel&> (s);
          moves_m.clear();
          for(size_t i=0; i < model.solution_.size(); i++)
          {
            if(model.solution_[i])
            {
              //edge is there, move will remove it
              //such a move is always possible!
              //moves_m.push_back (new move (i));

              //TODO: we can add here replace moves
            }
            else
            {
              //check first that the edge joining two points will not cause a cycle
              //by checking that the 2 vertices forming the edge are disconnected
              boost::graph_traits<Graph>::vertex_descriptor target = boost::target(model.edge_iterators_[i], model.LR_graph_);

              bool found = false;
              int target_v = static_cast<int>(target);
              bfs_time_visitor vis(found, target_v);
              boost::breadth_first_search(model.ST_, boost::source(model.edge_iterators_[i], model.LR_graph_), boost::visitor(vis));

              if(!vis.getFound())
              {
                moves_m.push_back (new move (i));
              }
            }
          }

          srand(time(NULL));
          std::random_shuffle (moves_m.begin (), moves_m.end ());
        }
      };

      int nviews_used_;

      //Given a solution (active edges):
        //maximize #edges used
        //minimize fsv_fraction
        //maximize overlap
      mets::gol_type
      evaluateSolution (const std::vector<bool> & active,
                          int changed,
                          std::vector< boost::graph_traits<Graph>::edge_descriptor> & edge_iterators,
                          Graph & ST_so_far_);

      void
      visualizeGlobalAlignment (bool visualize_cameras, std::string name = "global alignment");
    public:

      //parameters
      int max_poses_;

      ggo ();
      void
      readGraph (std::string & directory);
      void
      process ();
    };
  }
}

#endif /* GGO_H_ */
