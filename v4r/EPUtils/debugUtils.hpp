#ifndef EPDEBUG_H
#define EPDEBUG_H

#include "headers.hpp"

namespace EPUtils
{

/**
 * show image for debug
 * */
void showImageForDebug(std::string name, const cv::Mat &image, int waitTime = 5);
/**
 * print mat
 * */
void printMat(std::string name, cv::Mat &mat);
/**
 * print value
 * */
void printParameter(std::string name, float value);

// class to print
class DebugPrints
{
public:

  typedef boost::shared_ptr<DebugPrints> Ptr;
  typedef boost::shared_ptr<const DebugPrints> ConstPtr;
  
  typedef enum
  {
    NO_DEBUG_PRINT = 0,
    CRITICAL_PRINT = 1,
    LIGHT_TIME_PRINT = 2,
    FULL_TIME_PRINT = 3,
    FULL_DEBUG_PRINT = 4,
    RINGBUFFY_DEBUG_PRINT = 5
  } DEBUG_TYPE;

  ~DebugPrints();
  
  void setDebugType(int type);
  void print(const int type, const char *fmt, ...);
  void openFile(const char *fileName);
  void closeFile();

  //static DebugPrints::Ptr getDebug();

public:
  DebugPrints();
  
  int DebugType;
  FILE *pFile;
  bool fileOpened;

  boost::mutex debug_mutex;

  //static DebugPrints::Ptr debug;// = DebugPrints::Ptr(new DebugPrints());
};

//DebugPrints debug;

} //namespace EPUtils

#endif //EPDEBUG_H