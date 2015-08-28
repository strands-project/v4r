#include <algorithm>
#include <map>
#include <iostream>
#include <string>
#include <v4r/io/filesystem.h>

class Eval
{
public:
    class EvalPr
    {
    public:
        size_t tp_, fp_, fn_;
        std::string patrol_run;
        EvalPr()
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
    std::vector<EvalPr> eval_pr_;
    double f_score_;

    Eval()
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
};

struct less_f_score
{
    inline bool operator() (const Eval& eval1, const Eval& eval2)
    {
        return ( eval1.fscore()  < eval2.fscore() );
    }
};

int main(int argc, char ** argv)
{
    (void)argc;
    (void)argv;
    const std::string base_path = "/media/Data/datasets/icra16/eval_rec_online_models";
    const std::string test_base_path = "/media/Data/datasets/icra16/icra16_controlled_ba_test";
    std::map<std::string, size_t> views_per_run;
    std::vector<std::string> test_sub_folder;
    v4r::io::getFoldersInDirectory(test_base_path, "", test_sub_folder);
    for(size_t pr_id=0; pr_id<test_sub_folder.size(); pr_id++)
    {
        const std::string test_pr_path = test_base_path + "/" + test_sub_folder[pr_id];
        std::vector<std::string> file_rel;
        v4r::io::getFilesInDirectory(test_pr_path, file_rel, "", ".*.pcd", false);
        views_per_run[ test_sub_folder[pr_id] ] = file_rel.size();
    }

    std::vector<Eval> evals;

    std::vector<std::string> eval_sub_folder;
    v4r::io::getFoldersInDirectory(base_path,"",eval_sub_folder);
    for(size_t e_id=0; e_id<eval_sub_folder.size(); e_id++)
    {
        Eval e;
        e.path_ = base_path + "/" + eval_sub_folder[ e_id ];

        std::vector<std::string> patrol_run_folder;
        v4r::io::getFoldersInDirectory(e.path_,"",patrol_run_folder);

        for(size_t pr_id=0; pr_id<patrol_run_folder.size(); pr_id++)
        {
            const std::string prefix = e.path_ + "/" + patrol_run_folder[pr_id];

            std::vector<std::string> tested_views;
            v4r::io::getFoldersInDirectory(prefix,"",tested_views);
            for(size_t t_id=0; t_id<tested_views.size(); t_id++)
            {
                const std::string test_res_path = prefix + "/" + tested_views[t_id];
                Eval::EvalPr epr;
                epr.patrol_run = prefix;

                std::vector<std::string> file_rel;
                v4r::io::getFilesInDirectory(test_res_path, file_rel, "", ".*.txt", true);

                for(size_t r_id=0; r_id<file_rel.size(); r_id++)
                {
                    const std::string file = test_res_path + "/" + file_rel[ r_id ];
                    if( file.find("_patrol_run") != std::string::npos)
                    {
                        std::cout << file << "***************" << std::endl;
                        e.total_tp_++;
                    }
                    else
                    {
                        std::cout << file << std::endl;
                        e.total_fp_++;
                    }
                }

                size_t num_views = views_per_run[patrol_run_folder[pr_id]];
                epr.fn_ = num_views - epr.tp_;
                e.total_fn_ += epr.fn_;
                e.total_tp_ += epr.tp_;
                e.total_fp_ += epr.fp_;
                e.eval_pr_.push_back(epr);
            }
        }
        e.f_score_ = e.fscore();
        evals.push_back(e);
        std::cout << "====================================================================" << std::endl;
    }
    std::sort(evals.begin(), evals.end(), less_f_score());

    return 0;
}
