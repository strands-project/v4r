#include "SVMScale.h"

namespace svm
{

SVMScale::SVMScale()
{
  lower = -1.0;
  upper =  1.0;
  have_save_filename = false;
  have_features_filename = false;
  have_features_scaled_filename = false;
}

SVMScale::~SVMScale()
{
}

void SVMScale::compute()
{
  if(!have_features_filename)
  {
    fprintf(stderr,"do not have file with features\n");
    exit(1);
  }
  
  if(!(upper > lower))
  {
    fprintf(stderr,"inconsistent lower/upper specification\n");
    exit(1);
  }
  
  if(!have_save_filename)
  {
    fprintf(stderr,"do not have save_filename\n");
    exit(1);
  }
  
  if(!have_features_scaled_filename)
  {
    fprintf(stderr,"do not have have_features_scaled_filename\n");
    exit(1);
  }
	
  /* assumption: min index of attributes is 1 */
  /* pass 1: find out max index of attributes */
  int max_index = 0;
  int min_index = 1;

  std::ifstream fp1(features_filename.c_str());
  if( fp1.is_open() )
  {
    while( fp1.good() )
    {
      std::string line;
      getline(fp1,line);
      
      if(line.empty())
	break;
      
      std::stringstream ss;
      ss << line;
      
      int target;
      ss >> target;
      
      while(true)
      {
	std::string l;
	ss >> l;
	if(l.empty())
	  break;
	
	int index;
        double value;
        if(sscanf(l.c_str(),"%d:%lf",&index,&value)==2)
        {
	  if(index > max_index)
	    max_index = index;
        }
      }
    }
    fp1.close();
  }
  else
  {
    std::cerr << features_filename.c_str() << std::endl;
    fprintf(stderr,"cant open file.\n");
    exit(1);
  }
  
  if(min_index < 1)
    fprintf(stderr,"WARNING: minimal feature index is %d, but indices should start from 1\n", min_index);
  
  /* pass 2: find out max feature values of attributes */
  feature_max.clear();
  feature_max.resize(max_index+1);
  feature_min.clear();
  feature_min.resize(max_index+1);
  
  for(int i = 0; i <= max_index; ++i)
  {
    feature_max.at(i) = 0;
    feature_min.at(i) = 0;
  }
  
  bool firts_iteration = true;
  
  std::ifstream fp2(features_filename.c_str());
  if( fp2.is_open() )
  {
    while( fp2.good() )
    {
      std::string line;
      getline(fp2,line);
      
      if(line.empty())
	break;
      
      std::stringstream ss;
      ss << line;
      
      int target;
      ss >> target;
      
      while(true)
      {
	std::string l;
	ss >> l;
	if(l.empty())
	  break;
	
	int index;
        double value;
        if(sscanf(l.c_str(),"%d:%lf",&index,&value)==2)
        {
	  if(firts_iteration)
	  {
	    feature_max.at(index) = value;
	    feature_min.at(index) = value;
	  }
	  else
	  {
	    if(value < feature_min.at(index))
	      feature_min.at(index) = value;
	    if(value > feature_max.at(index))
	      feature_max.at(index) = value;
	  }
        }
      }
      firts_iteration = false;
    }
    fp2.close();
  }
  
  FILE *fp_save = fopen(save_filename.c_str(),"w");
  if(fp_save==NULL)
  {
    fprintf(stderr,"can't open file %s\n", save_filename.c_str());
    exit(1);
  }
  
  fprintf(fp_save, "x\n");
  fprintf(fp_save, "%.16g %.16g\n", lower, upper);
  for(int i=1; i <= max_index; i++)
  {
    if(feature_min.at(i) != feature_max.at(i))
      fprintf(fp_save,"%d %.16g %.16g\n",i,feature_min.at(i),feature_max.at(i));
  }
  
  if(min_index < 1)
    fprintf(stderr,"WARNING: scaling factors with indices smaller than 1 are not stored to the file %s.\n", save_filename.c_str());

  fclose(fp_save);
  
  // pass 3: scale
  std::ifstream fp3(features_filename.c_str());
  FILE *fp_scaled = fopen(features_scaled_filename.c_str(),"w");
  
  if(fp_scaled==NULL)
  {
    fprintf(stderr,"can't open file %s\n", features_scaled_filename.c_str());
    exit(1);
  }
  
  if( fp3.is_open() )
  {
    while( fp3.good() )
    {
      std::string line;
      getline(fp3,line);
      
      if(line.empty())
	break;
      
      std::stringstream ss;
      ss << line;
      
      int target;
      ss >> target;
      
      fprintf(fp_scaled, "%d ", target);
      
      while(true)
      {
	std::string l;
	ss >> l;
	if(l.empty())
	  break;
	
	int index;
        double value;
        if(sscanf(l.c_str(),"%d:%lf",&index,&value)==2)
        {
	  scale(index,value);
	  fprintf(fp_scaled,"%d:%lf ",index,value);
        }
      }
      fprintf(fp_scaled,"\n");
    }
    fp3.close();
  }
  fclose(fp_scaled);
}

void SVMScale::scale(int index, double &value)
{
  // skip single-valued attribute
  if(fabs(feature_max.at(index)-feature_min.at(index)) < 0.000001)
    return;

  if(fabs(value-feature_min.at(index)) < 0.000001)
    value = lower;
  else if(fabs(value-feature_max.at(index)) < 0.000001)
    value = upper;
  else
    value = lower + (upper-lower) * (value-feature_min.at(index))/(feature_max.at(index)-feature_min.at(index));
}

} //namespace svm