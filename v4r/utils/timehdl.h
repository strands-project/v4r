/***************************************************************************
 *   Copyright (C) 2010 by Markus Bader                                    *
 *   markus.bader@tuwien.ac.at                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
/**
 * @file timehdl.hpp
 * @author Markus Bader
 * @brief
 **/

#ifndef V4RTIMEHDL_H
#define V4RTIMEHDL_H

#include <sys/time.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>


namespace V4R {

/**
 * class to manage loop frequenses
 **/
class FQSleep {
public:
    FQSleep(const double &rFrequenz);
		void set(const double &rFrequenz);
    void wait();
private:
	  double mrFrequenz;
    boost::xtime mXTime;
    boost::xtime mPriod;
};

/**
 * class to manage timings
 **/
class Timer {
public:
    Timer() {};
    boost::posix_time::ptime  &start(const std::string &rName);
    boost::posix_time::time_duration &stop(const std::string &rName);
    boost::posix_time::time_duration &operator[](const std::string &rName) {
        return get(rName);
    }    
    boost::posix_time::time_duration &get(const std::string &rName) {
        return mDuration[rName];
    }
    long ms(const std::string &rName);
    std::string summary(bool allInOneLine = false, bool printUnits = true);
    std::list<std::string> labels();
private:
    std::map< std::string, boost::posix_time::ptime > mStartTime;
    std::map< std::string, boost::posix_time::time_duration > mDuration;
};

/**
 * class to manage thread timings
 * Like Timer, but uses clock_gettime(CLOCK_THREAD_CPUTIME_ID) to get the time
 * elapsed in the calling thread, rather than elapsed system time. The latter
 * can be arbitrarily longer, depending on system load.
 *
 * @author Michael Zillich
 **/
class ThreadTimer {
public:
    ThreadTimer() {};
    double start(const std::string &rName);
    double stop(const std::string &rName);
    double operator[](const std::string &rName) {
        return get(rName);
    }    
    double get(const std::string &rName) {
        return mDuration[rName];
    }
    std::string summary(bool allInOneLine = false);
private:
    std::map< std::string, double > mStartTime;
    std::map< std::string, double > mDuration;
};

/**
 * timeval to local ptime
 * @param sec
 * @param usec milliseconds
 * @param nsec nanoseconds
 * @return ptime
 **/
boost::posix_time::ptime  timeval2ptime_utc(unsigned long sec, unsigned long  usec, unsigned long  nsec = 0);
/**
 * timeval to local ptime
 * @param t
 * @return ptime
 **/
boost::posix_time::ptime  timeval2ptime_utc(const timeval &t);
/**
 * timeval to local ptime
 * This local adjustor depends on the machine TZ settings-- highly dangerous!
 * @param sec
 * @param usec milliseconds
 * @param nsec nanoseconds
 * @return ptime
 **/
boost::posix_time::ptime  timeval2ptime_local(unsigned long sec, unsigned long usec, unsigned long  nsec = 0);
/**
 * timeval to local ptime
 * This local adjustor depends on the machine TZ settings-- highly dangerous!
 * @param t
 * @return ptime
 **/
boost::posix_time::ptime  timeval2ptime_local(const timeval &t);

/** Sleep
* @param ms
*/
inline void sleepMs ( unsigned int ms ){
    boost::xtime xt;
    boost::xtime_get(&xt, boost::TIME_UTC);
    xt.nsec += ms * 1000 * 1000;
    boost::thread::sleep(xt);
}

/** Sleep
* @param sec
*/
inline void sleepSec ( double sec ){
    boost::xtime xt;
    boost::xtime_get(&xt, boost::TIME_UTC);
    xt.nsec +=  (sec  * 1000. * 1000. * 1000.);
    boost::thread::sleep(xt);
}

/** Boost ptime to timeval
* @param src
* @param des
*/
void ptime2timeval ( const boost::posix_time::ptime &src, timeval &des );

/** Boost ptime to sec & usec
* @param src
* @param sec_des
* @param usec_des
*/
void ptime2timeval ( const boost::posix_time::ptime &src, unsigned long &sec_des, unsigned long &usec_des);

/** Boost ptime to sec & usec
* @param src
* @param sec_des
* @param usec_des
*/
void ptime2timeval ( const boost::posix_time::ptime &src, uint64_t &sec_des, uint64_t &usec_des);

/** Boost ptime to sec & usec
* @param src
* @param sec_des
* @param usec_des
*/
void ptime2timeval ( const boost::posix_time::ptime &src, uint32_t &sec_des, uint32_t &usec_des);

/** Boost ptime to timeval
* @param src
* @param des
*/
timeval ptime2timeval ( const boost::posix_time::ptime &src);

};

#endif
