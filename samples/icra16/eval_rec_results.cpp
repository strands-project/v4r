#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <map>
#include <iostream>
#include <string>
#include <v4r/io/filesystem.h>

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
            return (double)tp_ / (tp_ + fn_);
        }

        double precision() const
        {
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

    TotalEval()
    {
        total_tp_ = total_fp_ = total_fn_ = num_prs_ = 0;
    }

    double recall() const
    {
        return (double)total_tp_ / (total_tp_ + total_fn_);
    }

    double precision() const
    {
        return (double)total_tp_ / (total_tp_ + total_fp_);
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
    (void)argc;
    (void)argv;
    const std::string base_path = "/media/Data/datasets/icra16/eval_rec_online_models";
    const std::string rec_results_online_unc = "/media/Data/datasets/icra16/eval_rec_online_models/sv_recognition_first_frame_0";
    const std::string base_path_offline = "/media/Data/datasets/icra16/results_sv_offline_models";
    const std::string test_base_path = "/media/Data/datasets/icra16/icra16_controlled_ba_test";
    const std::string map_file = "/media/Data/datasets/icra16/icra16_controlled_ba_test/object_list.csv";
    const std::string map_file_uncontrolled = "/media/Data/datasets/icra16/icra16_controlled_ba_test/object_list.csv";
    std::map<std::string, size_t> views_per_run;
    std::map<std::string, std::string> pr2obj;
    std::map<std::string, std::string> pr2obj_unc;
    std::map<std::string, std::vector<std::string> > obj2pr;
    std::map<std::string, std::vector<std::string> > obj2pr_unc;
    std::vector<std::string> test_sub_folder;
    v4r::io::getFoldersInDirectory(test_base_path, "", test_sub_folder);
    for(size_t pr_id=0; pr_id<test_sub_folder.size(); pr_id++)
    {
        const std::string test_pr_path = test_base_path + "/" + test_sub_folder[pr_id];
        std::vector<std::string> file_rel;
        v4r::io::getFilesInDirectory(test_pr_path, file_rel, "", ".*.pcd", false);
        views_per_run[ test_sub_folder[pr_id] ] = file_rel.size();
    }


    // map patrol run to objects for controlled runs
    std::ifstream in;
    in.open (map_file.c_str (), std::ifstream::in);
    char linebuf[1024];
    while(in.getline (linebuf, 1024))
    {
        std::string line (linebuf);
        std::vector < std::string > strs_2;
        boost::split (strs_2, line, boost::is_any_of (","));
        if (strs_2.size() > 2 && strs_2[2].length())
            continue;

        const std::string patrol_run_tmp = strs_2[0];
        const std::string obj = strs_2[1];
        pr2obj[patrol_run_tmp] = obj;

        std::map<std::string, std::vector<std::string> >::iterator it = obj2pr.find(obj);
        if (it != obj2pr.end() )
        {
            it->second.push_back(patrol_run_tmp);
        }
        else
        {
            std::vector<std::string> pr_tmp;
            pr_tmp.push_back(patrol_run_tmp);
            obj2pr[obj] = pr_tmp;
        }
    }
    in.close();

    // map patrol run to objects for uncontrolled runs
    in.open (map_file_uncontrolled.c_str (), std::ifstream::in);
    while(in.getline (linebuf, 1024))
    {
        std::string line (linebuf);
        std::vector < std::string > strs_2;
        boost::split (strs_2, line, boost::is_any_of (","));
        if (strs_2.size() > 2 && strs_2[2].length())
            continue;

        const std::string patrol_run_tmp = strs_2[0];
        const std::string obj = strs_2[1];
        pr2obj_unc[patrol_run_tmp] = obj;

        std::map<std::string, std::vector<std::string> >::iterator it = obj2pr_unc.find(obj);
        if (it != obj2pr_unc.end() )
        {
            it->second.push_back(patrol_run_tmp);
        }
        else
        {
            std::vector<std::string> pr_tmp;
            pr_tmp.push_back(patrol_run_tmp);
            obj2pr_unc[obj] = pr_tmp;
        }
    }
    in.close();

    std::map<std::string, TotalEval::Eval> object_based_eval;
    typedef std::map<std::string, std::vector<std::string> >::iterator it_type_2;
    for(it_type_2 it = obj2pr.begin(); it != obj2pr.end(); it++)
    {
        TotalEval::Eval eval_tmp;
        eval_tmp.eval_name_ = it->first;
        object_based_eval[it->first] = eval_tmp;
        // iterator->first = key
        // iterator->second = value
        // Repeat if you also want to iterate through the second map.
    }


    std::vector<TotalEval> online_evals;
    std::vector< std::map<std::string, TotalEval::Eval> > online_evals_obj_based;
    std::vector<std::string> eval_sub_folder;
    v4r::io::getFoldersInDirectory(base_path,"",eval_sub_folder);
    for(size_t e_id=0; e_id<eval_sub_folder.size(); e_id++)
    {
        TotalEval e;
        e.path_ = base_path + "/" + eval_sub_folder[ e_id ];

        if( e.path_.find("_100") == std::string::npos)
            continue;


        std::map<std::string, TotalEval::Eval> object_based_eval_tmp = object_based_eval;

        std::vector<std::string> patrol_run_folder;
        v4r::io::getFoldersInDirectory(e.path_,"",patrol_run_folder);

        for(size_t pr_id=0; pr_id<patrol_run_folder.size(); pr_id++)
        {
            const std::string prefix = e.path_ + "/" + patrol_run_folder[pr_id];
            TotalEval::Eval epr;
            epr.eval_name_ = prefix;
            epr.eval_id_ = patrol_run_folder[pr_id];
            boost::replace_all(epr.eval_id_, "patrol_run_", "");
            size_t num_views = 0;

            std::vector<std::string> tested_views;
            v4r::io::getFoldersInDirectory(prefix,"",tested_views);
            for(size_t t_id=0; t_id<tested_views.size(); t_id++)
            {
                const std::string test_res_path = prefix + "/" + tested_views[t_id];

                std::vector<std::string> file_rel;
                v4r::io::getFilesInDirectory(test_res_path, file_rel, "", ".*.txt", true);

                for(size_t r_id=0; r_id<file_rel.size(); r_id++)
                {
                    const std::string file = test_res_path + "/" + file_rel[ r_id ];
                    if( file.find("_patrol_run") != std::string::npos)
                    {
                        std::cout << file << "***************" << std::endl;
                        epr.tp_++;
                    }
                    else
                    {
                        std::cout << file << std::endl;
                        epr.fp_++;
                    }
                }

                num_views+= views_per_run[ tested_views[t_id] ];
            }
            if ( ((int)num_views - (int)epr.tp_) < 0)
                std::cout<< "WHAT???" << std::endl;
            epr.fn_ = num_views - epr.tp_;
            e.total_fn_ += epr.fn_;
            e.total_tp_ += epr.tp_;
            e.total_fp_ += epr.fp_;
            e.eval_pr_.push_back(epr);

            TotalEval::Eval &e_obj = object_based_eval_tmp[ pr2obj[patrol_run_folder[pr_id]] ];
            e_obj.fp_ += epr.fp_;
            e_obj.tp_ += epr.tp_;
            e_obj.fn_ += epr.fn_;
        }
        e.f_score_ = e.fscore();
        online_evals.push_back(e);
        online_evals_obj_based.push_back(object_based_eval_tmp);
        std::cout << "====================================================================" << std::endl;
    }
    std::sort(online_evals.begin(), online_evals.end(), less_f_score());

    online_evals.back().write2file();



    //-----------------offline eval---------------------

    TotalEval e_off;
    e_off.path_ = base_path_offline;

    typedef std::map<std::string, std::string>::iterator it_type;
    for(it_type it = pr2obj.begin(); it != pr2obj.end(); it++)
    {
        TotalEval::Eval epr;
        epr.eval_name_ = base_path_offline + "/" + it->first;
        epr.eval_id_ = it->first;
        boost::replace_all(epr.eval_id_, "patrol_run_", "");
        std::string gt_obj = it->second;
        boost::replace_all(gt_obj, ".pcd", "");

        std::vector<std::string> file_rel;
        v4r::io::getFilesInDirectory(epr.eval_name_, file_rel, "", ".*.txt", true);
        for(size_t r_id=0; r_id<file_rel.size(); r_id++)
        {
            if( file_rel[ r_id ].find(gt_obj) != std::string::npos)
            {
                std::cout << file_rel[ r_id ] << "***************" << std::endl;
                epr.tp_++;
            }
            else
            {
                std::cout << file_rel[ r_id ] << std::endl;
                epr.fp_++;
            }
        }

        size_t num_views = views_per_run[ it-> first ];
        epr.fn_ = num_views - epr.tp_;
        e_off.total_fn_ += epr.fn_;
        e_off.total_tp_ += epr.tp_;
        e_off.total_fp_ += epr.fp_;
        e_off.eval_pr_.push_back(epr);

        TotalEval::Eval &e_obj = object_based_eval[it->second];
        e_obj.fp_ += epr.fp_;
        e_obj.tp_ += epr.tp_;
        e_obj.fn_ += epr.fn_;

        // iterator->first = key
        // iterator->second = value
        // Repeat if you also want to iterate through the second map.
    }

    e_off.write2file("/tmp/offline_eval.txt");

    std::ofstream f;
    const std::string fn_off_obj_based = "/tmp/offline_eval_obj_based.txt";
    f.open (fn_off_obj_based.c_str());

    typedef std::map<std::string, TotalEval::Eval >::iterator it_type_3;
    for(it_type_3 it = object_based_eval.begin(); it != object_based_eval.end(); it++)
    {
        TotalEval::Eval &e = it->second;
        f << e.eval_name_ << " " << e.tp_ << " " << e.fp_ << " " << e.fn_ << std::endl;
        // iterator->first = key
        // iterator->second = value
        // Repeat if you also want to iterate through the second map.
    }
    f.close();

    const std::string fn_online_obj_based = "/tmp/online_eval_obj_based.txt";
    f.open (fn_online_obj_based.c_str());
    for(it_type_3 it = online_evals_obj_based[0].begin(); it != online_evals_obj_based[0].end(); it++)
    {
        TotalEval::Eval &e = it->second;
        f << e.eval_name_ << " " << e.tp_ << " " << e.fp_ << " " << e.fn_ << std::endl;
        // iterator->first = key
        // iterator->second = value
        // Repeat if you also want to iterate through the second map.
    }
    f.close();




    // for uncontrolled models
    std::map<std::string, TotalEval::Eval> object_based_eval_unc;
    typedef std::map<std::string, std::vector<std::string> >::iterator it_type_2;
    for(it_type_2 it = obj2pr_unc.begin(); it != obj2pr_unc.end(); it++)
    {
        TotalEval::Eval eval_tmp;
        eval_tmp.eval_name_ = it->first;
        object_based_eval_unc[it->first] = eval_tmp;
        // iterator->first = key
        // iterator->second = value
        // Repeat if you also want to iterate through the second map.
    }


    std::vector<std::string> patrol_run_folder;
    v4r::io::getFoldersInDirectory(rec_results_online_unc,"",patrol_run_folder);
    for(size_t pr_id=0; pr_id<patrol_run_folder.size(); pr_id++)
    {
        const std::string prefix = rec_results_online_unc + "/" + patrol_run_folder[pr_id];
        TotalEval::Eval epr;
        epr.eval_name_ = prefix;
        epr.eval_id_ = patrol_run_folder[pr_id];
        boost::replace_all(epr.eval_id_, "patrol_run_", "");
        size_t num_views = 0;

        std::vector<std::string> tested_views;
        v4r::io::getFoldersInDirectory(prefix,"",tested_views);
        for(size_t t_id=0; t_id<tested_views.size(); t_id++)
        {
            const std::string test_res_path = prefix + "/" + tested_views[t_id];

            std::vector<std::string> file_rel;
            v4r::io::getFilesInDirectory(test_res_path, file_rel, "", ".*.txt", true);

            for(size_t r_id=0; r_id<file_rel.size(); r_id++)
            {
                const std::string file = test_res_path + "/" + file_rel[ r_id ];
                if( file.find("_patrol_run") != std::string::npos)
                {
                    std::cout << file << "***************" << std::endl;
                    epr.tp_++;
                }
                else
                {
                    std::cout << file << std::endl;
                    epr.fp_++;
                }
            }

            num_views+= views_per_run[ tested_views[t_id] ];
        }
        if ( ((int)num_views - (int)epr.tp_) < 0)
            std::cout<< "WHAT???" << std::endl;
        epr.fn_ = num_views - epr.tp_;

        TotalEval::Eval &e_obj = object_based_eval_unc[ pr2obj_unc[patrol_run_folder[pr_id]] ];
        e_obj.fp_ += epr.fp_;
        e_obj.tp_ += epr.tp_;
        e_obj.fn_ += epr.fn_;
    }

    const std::string fn_online_obj_based_unc = "/tmp/online_eval_obj_based_unc.txt";
    f.open (fn_online_obj_based_unc.c_str());
    for(it_type_3 it = object_based_eval_unc.begin(); it != object_based_eval_unc.end(); it++)
    {
        TotalEval::Eval &e = it->second;
        f << e.eval_name_ << " " << e.tp_ << " " << e.fp_ << " " << e.fn_ << std::endl;
        // iterator->first = key
        // iterator->second = value
        // Repeat if you also want to iterate through the second map.
    }
    f.close();

    return 0;
}
