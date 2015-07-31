#ifndef V4R_COMMON_MACROS_H_
#define V4R_COMMON_MACROS_H_

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined V4RAPI_EXPORTS
#  define V4R_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define V4R_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define V4R_EXPORTS
#endif

#endif /* V4R_COMMON_MACROS_H_ */

