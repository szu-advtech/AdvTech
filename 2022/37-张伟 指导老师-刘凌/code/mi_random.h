#ifndef mi_random_h__
#define mi_random_h__

#include <random>
#include <functional>
 
/// A class to generate random values from multiple distributions
class MIRandom
{
protected:
	typedef  std::uniform_int_distribution<int>  uniform_int_dist;
	typedef  std::uniform_real_distribution<double>  uniform_double_dist;
	typedef  std::bernoulli_distribution  bernoulli_dist;
	typedef  std::exponential_distribution<double> exponential_dist;
	typedef  std::weibull_distribution<double> weibull_dist;

	typedef  std::mt19937  random_engine;		// Mersenne twister MT19937
	typedef  std::random_device device;

	random_engine engine;

public:
	MIRandom();

	// [a,b]
	int RandInt(int a, int b);
	double RandUnit(); 
	bool RandBernoulli(double p);
	double RandExp(double a);
	double RandWeibull(double a, double b);
};

/// To convert PDF and CDF from parameters
class PdfCdfConverter
{
public:
	static double ExpPDF(double x, double a);
	static double ExpCDF(double x, double a);
	static double WeibullPDF(double x, double a, double b);
	static double WeibullCDF(double x, double a, double b);
};


#endif // mi_random_h__
