#ifndef TIMEUTILS_H
#define TIMEUTILS_H

#include "headers.hpp"

namespace EPUtils
{
  
class TimeEstimationClass
{
private:
  bool isCounterStarted;
  bool isCounterWorkComplete;
    
  clockid_t clockID;
  timespec startTime, endTime;
    
public:
  TimeEstimationClass(clockid_t clockID = CLOCK_REALTIME);
  virtual ~TimeEstimationClass(){};
    
  void setClockID(clockid_t clockID);
  void countingStart();
  void countingEnd();
  
  unsigned long long getRealNanosecondsCount(timespec time);
  unsigned long long getCurrentTimeInNanoseconds();
  unsigned long getCurrentTimeInSeconds();
  unsigned long long getWorkTimeInNanoseconds();
  unsigned long getWorkTimeInSeconds();
};

} //namespace EPUtils

#endif //TIMEUTILS_H