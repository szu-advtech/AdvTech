
#ifndef INPUTPARAMETER_H_
#define INPUTPARAMETER_H_

#include <iostream>
#include <string>
#include <stdint.h>


using namespace std;

class InputParameter {
public:
	InputParameter();
	virtual ~InputParameter();

	/* Functions */
	void ReadInputParameterFromFile(const std::string & inputFile);
	void PrintInputParameter();
	//modified
	int capacity;
	int bank;
	int matrix;
	int cell_precision;
	int columnsubarray;
	int rowsubarray;
	string outputFilePrefix;
	int graphnodenumber;
	int graphedgenumber;
	//~modified
};

#endif /* INPUTPARAMETER_H_ */
