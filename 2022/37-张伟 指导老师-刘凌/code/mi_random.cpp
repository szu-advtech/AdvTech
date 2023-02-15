#include <iostream>
#include <cmath>
#include "mi_random.h"
 
MIRandom::MIRandom(void)
{
	device rd;
	engine.seed(rd());
}


double MIRandom::RandUnit()
{
	uniform_double_dist d(0.0, 1.0);
	return d(engine);
}


int MIRandom::RandInt(int a, int b)
{ 
	uniform_int_dist d(a, b);
	return d(engine);
}

bool MIRandom::RandBernoulli(double p)
{
	bernoulli_dist d(p);
	return d(engine);
}

double MIRandom::RandExp(double a)
{
	exponential_dist d(a);
	return d(engine);
}

double MIRandom::RandWeibull(double a, double b)
{
	weibull_dist d(a, b);
	return d(engine);
}


double PdfCdfConverter::ExpPDF(double x, double a)
{
	return a * exp(-a * x);
}

double PdfCdfConverter::ExpCDF(double x, double a)
{
	return 1.0 - exp(-a * x);
}

double PdfCdfConverter::WeibullPDF(double x, double a, double b)
{
	return (b/a) * pow(x/a, b-1) * exp(-pow(x/a, b));
}


double PdfCdfConverter::WeibullCDF(double x, double a, double b)
{
	return 1.0 - exp(-pow(x/a, b));
}
