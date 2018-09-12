#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include "fftw3.h"

#include <string>
#include <chrono>

#include <f2c.h>

#include <cuComplex.h>

#ifdef DOUBLEPRECISION
typedef cuDoubleComplex GPU_COMPLEX;
#else
typedef cuComplex GPU_COMPLEX;
#endif


#ifdef DOUBLEPRECISION
typedef double TYPE;
typedef doublecomplex F2C_COMPLEX;
typedef fftw_complex FFTW_COMPLEX;
#else
typedef float TYPE;
typedef complex F2C_COMPLEX;
typedef fftwf_complex FFTW_COMPLEX;
#endif


/*
 E N U M S
*/

enum class ComputationMode
{
	PREPROCESS,
	INTIME
};

// describes how image's boundaries are cut to avoid the so called 'ringing effect'
enum class ImageCut
{
	NoCut,
	CircleCut
};

enum class InputImageMode
{
	FromSource,
	FromObserved
};

enum class ImageFormat
{
	BMP,
	TXT,
	HDF5
};

enum class ColorType
{
	RGB,
	BW
};

enum class NoiseType
{
	Gaussian,
	Poisson
};

enum FileNameMode
{
	ZERO0,	// example 1, 20, 143
	ZERO3,	// example: 001, 020, 143
	ZERO4	// example: 0001, 0020, 0143
};

enum class Result
{
	Success,
	MemoryAllocationError,
	FileUnsupportedFormatError,
	FileUnsupportedNameModeError,
	FileOpenError,
	FileCreateError,
	FileReadError,
	FileWriteError,
	FileCloseError,
	FileDimensionError,
	MethodParametersError, 
	BLASMethodWrongArgument,
	BLASMethodZeroDiagonalElement,
	FFTWPlanCreatingError,
	CUDAMemoryAllocationError,
	CUDAMemoryCopyError,
	CUDAFFTPlanError,
	CUDAFFTExecutionError,
	CUDABLASInitializationError,
	UnsupportedDeviceType
};

enum class ExtensionMode
{
	Forward,
	Backward
};


/*
T Y P E D E F S
*/

typedef std::chrono::steady_clock::time_point timepoint;
typedef std::basic_ofstream<unsigned char, std::char_traits<unsigned char> > uofstream;
typedef std::basic_ifstream<unsigned char, std::char_traits<unsigned char> > uifstream;

/*
S T R U C T S
*/


struct ImageResolution
{
	unsigned int dimX;
	unsigned int dimY;
};

struct DataSettings
{
	std::string foldername;
	std::string filename;
	FileNameMode filename_mode;

	ImageResolution resolution;
};

struct OpticalSystemSettings
{
	TYPE depth;
	TYPE lambda;
	TYPE d1;
	TYPE d0;
	TYPE irisRadius;
	TYPE expFactor;
	OpticalSystemSettings(TYPE _depth, TYPE _lambda, TYPE _d1, TYPE _d0, TYPE _irisRadius) :
		depth(_depth), lambda(_lambda), d1(_d1), d0(_d0), irisRadius(_irisRadius), expFactor(M_PI * lambda * d1 * d1 / ((d0 + depth)*(d0 + depth))) {}
};

struct FocalDomainSettings
{
	TYPE minX;
	TYPE minY;
	TYPE maxX;
	TYPE maxY;
	FocalDomainSettings(TYPE _minX, TYPE _minY, TYPE _maxX, TYPE _maxY) :
		minX(_minX), minY(_minY), maxX(_maxX), maxY(_maxY) {}
};

struct ImageSettings
{
	ImageFormat format;
	ImageCut cut;
	ImageResolution resolution;
};

struct NoiseSettings
{
	NoiseType type;
	TYPE value;
};

struct MethodParameters
{
	unsigned int iterationsNumber[3];
	TYPE mu[3];
};

struct Image
{
	TYPE* data;
	ImageSettings settings;
};

struct ExtendedImage
{
	FFTW_COMPLEX* data;
	ImageResolution resolution;
};

/*
F U N C T I O N S
*/

std::string to_string(Result result);
std::string to_string(ImageFormat imageFormat);
std::string to_string(ComputationMode mode);

std::string get_full_filename(DataSettings file_settings, int i);
std::string get_full_filename(DataSettings file_settings, int chunk_number, int k);
std::string get_full_txt_filename(DataSettings file_settings, int k);

#define _react_on(result) if (result != Result::Success) { getchar(); return result;} 


/*
B M P   T Y P E D E F S
M U S T   B E   C A R E F U L L Y   C H E C K E D ! ! ! 
*/
#pragma pack(1)
#pragma once

typedef int LONG;
typedef unsigned short WORD;
typedef unsigned int DWORD;

typedef struct tagBITMAPFILEHEADER {
	WORD bfType;
	DWORD bfSize;
	WORD bfReserved1;
	WORD bfReserved2;
	DWORD bfOffBits;
} BITMAPFILEHEADER, *PBITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
	DWORD biSize;
	LONG biWidth;
	LONG biHeight;
	WORD biPlanes;
	WORD biBitCount;
	DWORD biCompression;
	DWORD biSizeImage;
	LONG biXPelsPerMeter;
	LONG biYPelsPerMeter;
	DWORD biClrUsed;
	DWORD biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER;

#endif