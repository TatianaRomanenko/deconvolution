#include "assert.h"
#include "typedefs.h"
#include "log.h"

#include <string>


std::string to_string(Result result)
{
	switch (result){
	case Result::FileCreateError:
		return "File creation failed.";
	case Result::FileDimensionError:
		return "File dimension mismatched."; 
	case Result::FileUnsupportedFormatError:
		return "Unsupported file format.";
	case Result::FileUnsupportedNameModeError:
		return "Unsupported filename mode.";
	case Result::FileOpenError:
		return "File opening failed.";
	case Result::FileReadError:
		return "File reading failed.";
	case Result::MemoryAllocationError:
		return "Memory allocation failed.";
	case Result::MethodParametersError:
		return "Wrong method parameters.";
	case Result::Success:
		return "Method succedded.";
	case Result::BLASMethodWrongArgument:
		return "BLAS method wrong argument passed.";
	case Result::BLASMethodZeroDiagonalElement:
		return "Blas method got zero U(i,i) in LU factorization.";
	case Result::UnsupportedDeviceType:
		return "Unsupported device type selected.";
	default:
		return "UNSUPPORTED Result value";
	}
}

std::string to_string(ImageFormat imageFormat)
{
	switch (imageFormat)
	{
	case ImageFormat::BMP:
		return ".bmp";
	case ImageFormat::TXT:
		return ".txt";
	case ImageFormat::HDF5:
		return ".h5";
	default:
		return "UNKNOWN IMAGE FORMAT";
	}
}


std::string to_string(ComputationMode mode)
{
	switch(mode)
	{
	case ComputationMode::INTIME:
		return "INTIME";
	case ComputationMode::PREPROCESS:
		return "PREPROCESS";
	default:
		return "UNKNOWN COMPUTATION MODE";
	}
}

std::string get_full_filename(DataSettings file_settings, int k){
	std::string result = "";
	std::string number_to_str = "";
	switch (file_settings.filename_mode){
	case FileNameMode::ZERO0:
		number_to_str = std::to_string(k);
		break;
	case FileNameMode::ZERO3:
		char buf[4];
		sprintf_s(buf, 4, "%03d", k);
		number_to_str += buf;
		break;
	case FileNameMode::ZERO4:
		char buf2[5];
		sprintf_s(buf2, 5, "%04d", k);
		number_to_str += buf2;
		break;
	default:
		log_operation_result("Unsupported name mode selected.", Result::FileUnsupportedNameModeError);
		break;
	}
	// TODO: Implement cross-platform solution
	// WARNING: Windows-only
	result = file_settings.foldername + "\\" + file_settings.filename + number_to_str + ".bin";
	return result;
}

std::string get_full_filename(DataSettings file_settings, int chunk_number, int k)
{
	std::string result = "";
	std::string chunk_number_to_str = std::to_string(chunk_number);
	std::string number_to_str = "";
	switch (file_settings.filename_mode){
	case FileNameMode::ZERO0:
		number_to_str = std::to_string(k);
		break;
	case FileNameMode::ZERO3:
		char buf[4];
		sprintf_s(buf, 4, "%03d", k);
		number_to_str += buf;
		break;
	case FileNameMode::ZERO4:
		char buf2[5];
		sprintf_s(buf2, 5, "%04d", k);
		number_to_str += buf2;
		break;
	default:
		log_operation_result("Unsupported name mode selected.", Result::FileUnsupportedNameModeError);
		break;
	}
	// TODO: Implement cross-platform solution
	// WARNING: Windows-only
	result = file_settings.foldername + "\\" + file_settings.filename + "_" + chunk_number_to_str + "_" + number_to_str + ".bin";
	return result;
}

std::string get_full_txt_filename(DataSettings file_settings, int k){
	std::string result = "";
	std::string number_to_str = "";
	switch (file_settings.filename_mode){
	case FileNameMode::ZERO0:
		number_to_str = std::to_string(k+1);
		break;
	case FileNameMode::ZERO3:
		char buf[4];
		sprintf_s(buf, 4, "%03d", k+1);
		number_to_str += buf;
		break;
	case FileNameMode::ZERO4:
		char buf2[5];
		sprintf_s(buf2, 5, "%04d", k+1);
		number_to_str += buf2;
		break;
	default:
		log_operation_result("Unsupported name mode selected.", Result::FileUnsupportedNameModeError);
		break;
	}
	// TODO: Implement cross-platform solution
	// WARNING: Windows-only
	result = file_settings.foldername + "\\" + file_settings.filename + number_to_str + ".txt";

	return result;
}