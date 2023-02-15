//  mi_util.h
//  max_influence

#ifndef mi_util_h__
#define mi_util_h__

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <exception>
#include <cmath>
#include <cassert>

/// Null Pointer Exceptions
class NullPointerException
    : public std::runtime_error
{
public:
    NullPointerException(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

class InvalidInputFormatException
	: public std::runtime_error
{
public:
	InvalidInputFormatException(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

#endif // mi_util_h__
