#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <map>
#include <numeric>
#include <iostream>
#include <string>
#include <v4r/io/filesystem.h>

namespace po = boost::program_options;

class TotalEval
{
public:
    class Eval
    {
    public:
        size_t tp_, fp_, fn_;
        std::string eval_name_, eval_id_;

        Eval()
        {
            tp_=0, fp_=0, fn_=0;
        }
        double recall() const
        {
            if( (tp_ + fn_) == 0)
                return 1;

            return (double)tp_ / (tp_ + fn_);
        }

        double precision() const
        {
            if( (tp_ + fp_) == 0)
                return -1;

            return (double)tp_ / (tp_ + fp_);
        }

        double fscore() const
        {
            return 2*precision()*recall() / (precision() + recall());
        }
    };

    std::string path_;
    size_t total_tp_, total_fp_, total_fn_;
    size_t num_prs_;
    std::vector<Eval> eval_pr_;
    double f_score_;

    size_t num_visible_pts_, num_total_pts_, num_trained_views_, num_total_views_;

    TotalEval()
    {
        total_tp_ = total_fp_ = total_fn_ = num_prs_ = 0;
    }

    double recall() const
    {
        size_t tp=0, fp=0, fn=0;

        for(size_t i=0; i<eval_pr_.size(); i++)
        {
            const TotalEval::Eval &e = eval_pr_[i];
            tp += e.tp_;
            fp += e.fp_;
            fn += e.fn_;
        }

        if ((tp + fn) == 0)
            return -1;

        return (double)tp / (tp + fn);
    }

    double precision() const
    {
        size_t tp=0, fp=0, fn=0;

        for(size_t i=0; i<eval_pr_.size(); i++)
        {
            const TotalEval::Eval &e = eval_pr_[i];
            tp += e.tp_;
            fp += e.fp_;
            fn += e.fn_;
        }

        if ((tp + fp) == 0)
            return -1;

        return (double)tp / (tp + fp);
    }

    double fscore() const
    {
        return 2*precision()*recall() / (precision() + recall());
    }

    void write2file(const std::string &filename = std::string("/tmp/eval.txt")) const
    {
        std::ofstream f;
        f.open (filename.c_str());
        for (size_t i=0; i < eval_pr_.size(); i++)
        {
            const Eval &e = eval_pr_[i];
            f << e.eval_id_ << " " << e.tp_ << " " << e.fp_ << " " << e.fn_ << std::endl;
        }
        f.close();
    }

    double avg_recall () const
    {
        std::vector<double> vals;
        for(size_t i=0; i<eval_pr_.size(); i++)
        {
            double val = eval_pr_[i].recall();

            if( val>=0.f )
                vals.push_back(val);
        }

        double sum = std::accumulate(vals.begin(), vals.end(), 0.f, std::plus<double>());
        return sum / vals.size();
    }

    double avg_precision () const
    {
        std::vector<double> vals;
        for(size_t i=0; i<eval_pr_.size(); i++)
        {
            double val = eval_pr_[i].precision();

            if( val>=0.f )
                vals.push_back(val);
        }

        double sum = std::accumulate(vals.begin(), vals.end(), 0.f, std::plus<double>());
        return sum / vals.size();
    }
};

struct less_f_score
{
    inline bool operator() (const TotalEval& eval1, const TotalEval& eval2)
    {
        return ( eval1.fscore()  < eval2.fscore() );
    }
};

int main(int argc, char ** argv)
{

    std::string anno_gt = "/media/Data/datasets/icra16/gt_annotations_new";
    std::string models_dir = "/media/Data/datasets/icra16/models";
    std::string rec_results = "/home/thomas/icra_rec_partial/cereal_box.pcd";
    std::string test_set_root_dir = "/media/Data/datasets/icra16/keyframes/controlled_ba";
    std::string out_file = "/tmp/eval_partial_model_rec_results.txt";

    po::options_description desc("Evaluation of partial model recognition results\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("anno_gt,g", po::value<std::string>(&anno_gt)->default_value(anno_gt), "")
            ("rec_results,r", po::value<std::string>(&rec_results)->default_value(rec_results), "")
            ("models_dir,m", po::value<std::string>(&models_dir)->default_value(models_dir), "")
            ("test_set_root_dir,t", po::value<std::string>(&test_set_root_dir)->default_value(test_set_root_dir), "")
  ;
     po::variables_map vm;
     po::store(po::parse_command_line(argc, argv, desc), vm);
     if (vm.count("help"))
     {
         std::cout << desc << std::endl;
         return false;
     }

     try
     {
         po::notify(vm);
     }
     catch(std::exception& e)
     {
         std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
         return false;
     }

     std::vector<std::string> models;
     v4r::io::getFilesInDirectory(models_dir, models, "", ".*.pcd", false);
     for(size_t m_id=0; m_id < models.size(); m_id++)
     {
         boost::replace_last(models[m_id], ".pcd", "");
     }

    std::vector<std::string> eval_ids;
    v4r::io::getFoldersInDirectory(rec_results, "", eval_ids);
    std::vector<TotalEval> evals(eval_ids.size());

    std::ofstream f(out_file.c_str());

    for(size_t e_id=0; e_id<eval_ids.size(); e_id++)
    {
        const std::string eval_fn = rec_results + "/" + eval_ids[e_id];

        std::ifstream info_f ( (eval_fn + "/model_info.txt"));
        std::string model_foo;

        TotalEval te;
        te.path_ = eval_fn;
        info_f >> model_foo >> te.num_trained_views_ >> te.num_total_views_ >> te.num_visible_pts_ >> te.num_total_pts_;
        info_f.close();

        std::vector<std::string> test_runs;
        v4r::io::getFoldersInDirectory(eval_fn, "", test_runs);
        for(size_t r_id=0; r_id<test_runs.size(); r_id++)
        {
            const std::string or_set = te.path_ + "/" + test_runs[r_id];
            const std::string test_set = test_set_root_dir + "/" + test_runs[r_id];
            const std::string anno_gt_set = anno_gt + "/" + test_runs[r_id];

            std::vector<std::string> views;
            v4r::io::getFilesInDirectory(test_set, views, "", ".*.pcd", true);

            for(size_t v_id=0; v_id < views.size(); v_id++)
            {
                std::string cloud_name (views[v_id]);
                boost::replace_last(cloud_name, ".pcd", "");
                TotalEval::Eval e;

                for(size_t m_id=0; m_id<models.size(); m_id++)
                {
                    const std::string anno_search_pattern = ".*" + cloud_name + "_" + models[m_id] + ".*.txt";

                    std::vector<std::string> annos_gt, annos_or;
                    v4r::io::getFilesInDirectory(anno_gt_set, annos_gt, "", anno_search_pattern, false);
                    v4r::io::getFilesInDirectory(or_set, annos_or, "", anno_search_pattern, false);

                    e.fp_ += std::max<int>(0, (int)annos_or.size() - (int)annos_gt.size());
                    e.fn_ += std::max<int>(0, (int)annos_gt.size() - (int)annos_or.size());
                    e.tp_ += std::min<int>(annos_gt.size(), annos_or.size());

                    int a=1;
                    a;
                }
                te.eval_pr_.push_back(e);


//                for(size_t a_id=0; a_id < annos_gt.size(); a_id++)
//                {
//                    std::string occlusion_pattern (views[v_id]);
//                    boost::replace_last(occlusion_pattern, ".pcd", "_occlusion_");

//                    if(annos_gt[a_id].compare(0, occlusion_pattern.length(), occlusion_pattern) == 0) // ignore occlusion files
//                        continue;

//                    std::string anno_file = anno_gt_set + "/" + annos_gt[a_id];
//                    std::string anno_occlusion_file = annos_gt[a_id];
//                    boost::replace_first(anno_occlusion_file, cloud_name, cloud_name + "_occlusion");
//                    std::ifstream occ_f ( (anno_gt_set + "/" + anno_occlusion_file).c_str());
//                    float occlusion_value;
//                    occ_f >> occlusion_value;
//                    occ_f.close();

//                    if (occlusion_value > .95)
//                        continue;

//                    std::vector<std::string> string_parts;
//                    boost::split (string_parts, annos_gt[a_id], boost::is_any_of ("_"));
//                    std::string search_pattern = string_parts[0];
//                    for(size_t s_id=1; s_id < string_parts.size()-1; s_id++)
//                        search_pattern += "_" + string_parts[s_id];

//                    bool found = false;
//                    for (size_t or_id=0; or_id < annos_or.size(); or_id++)
//                    {
//                        found=true;
//                    }
//                }

            }


//            TotalEval::Eval epr;
//            epr.eval_name_ = prefix;
//            epr.eval_id_ = test_runs[r_id];
//            boost::replace_all(epr.eval_id_, "patrol_run_", "");
//            size_t num_views = 0;

//            std::vector<std::string> views;
//            v4r::io::getFoldersInDirectory(prefix,"",views);
//            for(size_t t_id=0; t_id<views.size(); t_id++)
//            {
//                const std::string test_res_path = prefix + "/" + views[t_id];

//                std::vector<std::string> file_rel;
//                v4r::io::getFilesInDirectory(test_res_path, file_rel, "", ".*.txt", true);

//                for(size_t r_id=0; r_id<file_rel.size(); r_id++)
//                {
//                    const std::string file = test_res_path + "/" + file_rel[ r_id ];
//                    if( file.find("_patrol_run") != std::string::npos)
//                    {
//                        std::cout << file << "***************" << std::endl;
//                        epr.tp_++;
//                    }
//                    else
//                    {
//                        std::cout << file << std::endl;
//                        epr.fp_++;
//                    }
//                }

//                num_views+= views_per_run[ views[t_id] ];
//            }
//            if ( ((int)num_views - (int)epr.tp_) < 0)
//                std::cout<< "WHAT???" << std::endl;
//            epr.fn_ = num_views - epr.tp_;
//            te.total_fn_ += epr.fn_;
//            te.total_tp_ += epr.tp_;
//            te.total_fp_ += epr.fp_;
//            te.eval_pr_.push_back(epr);

//            TotalEval::Eval &e_obj = object_based_eval_tmp[ pr2obj[test_runs[r_id]] ];
//            e_obj.fp_ += epr.fp_;
//            e_obj.tp_ += epr.tp_;
//            e_obj.fn_ += epr.fn_;
//        }
//        te.f_score_ = te.fscore();
//        online_evals.push_back(te);
//        online_evals_obj_based.push_back(object_based_eval_tmp);
//        std::cout << "====================================================================" << std::endl;
        }
        evals[e_id] = te;

        std::cout << te.num_trained_views_ << " " << te.num_total_views_ << " " <<
                     te.num_visible_pts_ << " " << te.num_total_pts_ << " " <<
                     static_cast<float>(te.num_visible_pts_)/te.num_total_pts_ << " " << te.avg_precision()  <<
                     " " << te.avg_recall() << " " << te.precision() << " " << te.recall() << std::endl;

        f << te.num_trained_views_ << " " << te.num_total_views_ << " " <<
                     te.num_visible_pts_ << " " << te.num_total_pts_ << " " <<
                     static_cast<float>(te.num_visible_pts_)/te.num_total_pts_ << " " << te.avg_precision()  <<
                     " " << te.avg_recall() << " " << te.precision() << " " << te.recall() << std::endl;

    }
    f.close();

    return 0;
}
