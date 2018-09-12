#define _CRT_SECURE_NO_WARNINGS
#include "log.h"

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>

#include "typedefs.h"


char * __cdecl get_current_time();

void write_performance_header(std::string filename)
{
	std::ofstream out_file(filename, std::ios::out | std::ios::app);
	if (out_file.is_open()){
		out_file << "time" << ";" << "current_time" << ";" << "image_number" << ";" << "image_size" << ";" 
			<< "iterations" << ";" << "device" << ";" << "chunk_number" << "\n";  
		out_file.close();
	}
	else{
		log_operation_result("File " + filename + "opening failed for header saving", Result::FileOpenError);
	}
}

void write_performance_result(std::string filename, double seconds, int imageNumber, int imageSize, int iterations, int chunkNumber = 1)
{
	std::ofstream out_file(filename, std::ios::out | std::ios::app);
#ifdef CPU
	std::string device = "CPU";
#elif defined GPU
	std::string device = "GPU";
#endif
	if (out_file.is_open()){
		out_file << seconds << ";" 
			//<< get_current_time() << ";" 
			<< imageNumber << ";" << imageSize << ";" 
			<< iterations << ";" << device << ";"  << chunkNumber << "\n";  
		out_file.close();
	}
	else{
		log_operation_result("File " + filename + "opening failed for results saving", Result::FileOpenError);
	}
}


void log_message(std::string message)
{
	std::cout  << get_current_time() << message << "\n\n";
}

Result log_operation_result(std::string message, Result op_result)
{
	std::cout << get_current_time()  << message <<
		((op_result == Result::Success) ? "succeeded.\n\n" : ("failed with the following error: " + to_string(op_result) + ".\n\n"));
	return op_result;
}

void log_measured_event(std::string message, timepoint start, timepoint end)
{
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout <<  get_current_time() << message << " Elapsed time: " << elapsed_seconds.count() << "s.\n\n";
}

char * __cdecl get_current_time()
{
	std::chrono::time_point<std::chrono::system_clock> current_time_point = std::chrono::system_clock::now();
	std::time_t current_time = std::chrono::system_clock::to_time_t(current_time_point);
	return std::ctime(&current_time);
}