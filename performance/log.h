#ifndef LOG_H 
#define LOG_H

#include <string>

#include "typedefs.h"


/*
Current realization of logging functions uses only std::cout
file logging support in progress...
*/

// saves result of computation method
void write_performance_header(std::string filename);

void write_performance_result(std::string filename, double seconds, int imageNumber, int imageSize, int iterations, int chunkNumber);

// write log message to the selected log file
void log_message(std::string message);

Result log_operation_result(std::string message, Result op_result);

void log_measured_event(std::string message, timepoint start, timepoint end);

char * __cdecl get_current_time();

#endif