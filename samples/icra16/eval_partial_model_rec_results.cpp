#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <map>
#include <numeric>
#include <iostream>
#include <string>
#include <v4r/io/filesystem.h>


//#define TURNTABLE

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
        f << num_trained_views_ << " " << num_total_views_ << " " << num_visible_pts_ << " " << num_total_pts_ << " " <<
             static_cast<float>(num_visible_pts_)/num_total_pts_ << " " << avg_precision()  <<
             " " << avg_recall() << " " << precision() << " " << recall() << std::endl;
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

    std::string anno_gt = "/home/thomas/Documents/icra16/gt_annotations";
    std::string models_dir = "/home/thomas/Documents/icra16/turntable_models/models";
#ifdef TURNTABLE
    std::string rec_results = "/home/thomas/Documents/icra16/test_results/rec_rate_wrt_coverage/turntable";
    std::string out_dir = "/home/thomas/Documents/icra16/test_results/rec_rate_wrt_coverage_eval/turntable";
#else
//    std::string rec_results = "/home/thomas/Documents/icra16/test_results/rec_rate_wrt_coverage/controlled";
//    std::string out_dir = "/home/thomas/Documents/icra16/test_results/rec_rate_wrt_coverage_eval/controlled";
    std::string rec_results = "/home/thomas/Documents/icra16/test_results/rec_rate_wrt_coverage/uncontrolled";
    std::string out_dir = "/home/thomas/Documents/icra16/test_results/rec_rate_wrt_coverage_eval/uncontrolled";
#endif
    std::string test_set_root_dir = "/home/thomas/Documents/icra16/keyframes/controlled_ba";

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


    boost::filesystem::remove_all(boost::filesystem::path(out_dir));

    std::vector<std::string> models;
    v4r::io::getFilesInDirectory(models_dir, models, "", ".*.pcd", false);

    for(size_t m_id=0; m_id < models.size(); m_id++)
        boost::replace_last(models[m_id], ".pcd", "");

    std::vector<std::string> eval_sets;
    v4r::io::getFoldersInDirectory(rec_results, "", eval_sets);

    for(size_t set_id=0; set_id<eval_sets.size(); set_id++)
    {
        const std::string eval_path = rec_results + "/" + eval_sets[set_id];
#ifndef TURNTABLE
        std::vector<std::string> runs;
        v4r::io::getFoldersInDirectory(eval_path, "", runs);

        for(size_t lr_id=0; lr_id<runs.size(); lr_id++)
        {
#endif

        std::vector<std::string> eval_ids;
#ifdef TURNTABLE
        const std::string out_fn = out_dir + "/" + eval_sets[set_id] + "/rec_rate_wrt_coverage_result.txt";
        v4r::io::getFoldersInDirectory(eval_path, "", eval_ids);
#else
        v4r::io::getFoldersInDirectory(eval_path + "/" + runs[lr_id], "", eval_ids);
        const std::string out_fn = out_dir + "/" + eval_sets[set_id] + "/" + runs[lr_id] + "/rec_rate_wrt_coverage_result.txt";
#endif
        std::vector<TotalEval> evals(eval_ids.size());

        v4r::io::createDirForFileIfNotExist(out_fn);
        std::ofstream f(out_fn.c_str());

        for(size_t e_id=0; e_id<eval_ids.size(); e_id++)
        {
#ifdef TURNTABLE
            const std::string eval_fn = eval_path + "/" + eval_ids[e_id];
#else
            const std::string eval_fn = eval_path + "/" + runs[lr_id] + "/" + eval_ids[e_id];
#endif

            std::ifstream info_f ( (eval_fn + "/model_info.txt"));
            std::string current_model;

            TotalEval te;
            te.path_ = eval_fn;
            info_f >> current_model >> te.num_trained_views_ >> te.num_total_views_ >> te.num_visible_pts_ >> te.num_total_pts_;
            info_f.close();

            boost::replace_last(current_model, ".pcd", "");

            std::vector<std::string> test_runs;
            v4r::io::getFoldersInDirectory(eval_fn, "", test_runs);
            for(size_t r_id=0; r_id<test_runs.size(); r_id++)
            {
#ifndef TURNTABLE
                if(test_runs[r_id] == runs[lr_id])  // don't test on trained sequence
                    continue;
#endif
                const std::string or_set = te.path_ + "/" + test_runs[r_id];
                const std::string test_set = test_set_root_dir + "/" + test_runs[r_id];
                const std::string anno_gt_set = anno_gt + "/" + test_runs[r_id];

                std::vector<std::string> views;

                if(!v4r::io::existsFile(te.path_ + "/" + test_runs[r_id] + "/taken_views.nfo"))
                {
                    v4r::io::getFilesInDirectory(test_set, views, "", ".*.pcd", true);
                }
                else
                {
                    std::ifstream tv_f((te.path_ + "/" + test_runs[r_id] + "/taken_views.nfo").c_str());
                    std::string line;
                    while (std::getline(tv_f, line))
                        views.push_back(line);
                }


                for(size_t v_id=0; v_id < views.size(); v_id++)
                {
                    std::string cloud_name (views[v_id]);
                    boost::replace_last(cloud_name, ".pcd", "");
                    TotalEval::Eval e;

                    for(size_t m_id=0; m_id<models.size(); m_id++)
                    {
                        const std::string anno_search_pattern = ".*" + cloud_name + "_" + models[m_id] + ".*.txt";

                        std::vector<std::string> annos_gt, annos_or;

                        if(!v4r::io::existsFolder(anno_gt_set)) // if there are no annotations for this run, just skip
                            continue;

                        v4r::io::getFilesInDirectory(anno_gt_set, annos_gt, "", anno_search_pattern, false);

#ifdef TURNTABLE
                        v4r::io::getFilesInDirectory(or_set, annos_or, "", anno_search_pattern, false);
#else
                        if(current_model.compare(models[m_id]) == 0)
                        {
                            const std::string anno_search_pattern_learnt = ".*" + cloud_name + "_" + runs[lr_id] + ".*.txt";
                            v4r::io::getFilesInDirectory(or_set, annos_or, "", anno_search_pattern_learnt, false);
                        }
                        else
                        {
                            v4r::io::getFilesInDirectory(or_set, annos_or, "", anno_search_pattern, false);
                        }
#endif

                        size_t fn = std::max<int>(0, (int)annos_gt.size() - (int)annos_or.size());
                        if(fn>0) // check if the object was actually visible
                        {
                            const std::string anno_occlusion_search_pattern = ".*" + cloud_name + "_occlusion_" + models[m_id] + ".*.txt";
                            std::vector<std::string> annos_occ_gt;
                            v4r::io::getFilesInDirectory(anno_gt_set, annos_occ_gt, "", anno_occlusion_search_pattern, false);

                            for(size_t o=0; o<annos_occ_gt.size(); o++)
                            {
                                std::ifstream occ_s( (anno_gt_set + "/" + annos_occ_gt[o]).c_str());
                                float occ;
                                occ_s >> occ;
                                occ_s.close();

                                if(occ>0.95)    // if the object is occluded by more than 95% neglect it
                                    fn--;
                            }
                        }

                        e.fp_ += std::max<int>(0, (int)annos_or.size() - (int)annos_gt.size());
                        e.fn_ += std::max<int>(0, fn);
                        e.tp_ += std::min<int>(annos_gt.size(), annos_or.size());
                    }
                    te.eval_pr_.push_back(e);
                }
                evals[e_id] = te;
            }
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

#ifndef TURNTABLE
        }
#endif
    }

    return 0;
}
