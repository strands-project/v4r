libFreenectIncludeDir = '[Path to LibFreenect Include directory]';
eval(sprintf('mex -I%s get_accel_data.cpp', libFreenectIncludeDir));

