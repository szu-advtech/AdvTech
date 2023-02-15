
#include "PIM_InputParameter.h"
//#include "constant.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

InputParameter::InputParameter() {
	//modified by Han
	capacity=16;
	bank=8;
	matrix=4;  //It has to the multiple times of 2, because 2 mats share one senseamp
	cell_precision=8;
	columnsubarray=256;
	rowsubarray=256;
	// TODO Auto-generated constructor stub
	graphnodenumber=0;
	graphedgenumber=0;
	outputFilePrefix = "output";	/* Default output file name */
}

InputParameter::~InputParameter() {
	// TODO Auto-generated destructor stub
}

void InputParameter::ReadInputParameterFromFile(const std::string & inputFile) {
	FILE *fp = fopen(inputFile.c_str(), "r");
	char line[500];
	char tmp[500];
	
	if (!fp) {
		cout << inputFile << " cannot be found!\n";
		exit(-1);
	}

	while (fscanf(fp, "%[^\n]\n", line) != EOF) {
	
		 if (!strncmp("-Capacity", line, strlen("-Capacity"))) 
		{  
                      sscanf(line, "-Capacity: %d", &capacity);
                        continue;
                }
		if (!strncmp("-Bank", line, strlen("-Bank"))) 
                {                        
                      sscanf(line, "-Bank: %d", &bank);           
			 continue;
                }

		if (!strncmp("-Matrix", line, strlen("-Matrix"))) 
                {                        
                      sscanf(line, "-Matrix: %d", &matrix);    
	             continue;
                }


		 if (!strncmp("-Cellprecision", line, strlen("-Cellprecision")))
                {
                      sscanf(line, "-Cellprecision: %d", &cell_precision);
                     continue;
                }

		  if (!strncmp("-Columnsubarray", line, strlen("-Columnsubarray")))                {
                      sscanf(line, "-Columnsubarray: %d", &columnsubarray);
                     continue;
                }

		  if (!strncmp("-Rowsubarray", line, strlen("-Rowsubarray")))                {
                      sscanf(line, "-Rowsubarray: %d", &rowsubarray);
                     continue;
                }
	

		if (!strncmp("-OutputFilePrefix", line, strlen("-OutputFilePrefix"))) {
			sscanf(line, "-OutputFilePrefix: %s", tmp);
			outputFilePrefix = (string)tmp;
			continue;
		}
	

		if (!strncmp("-GraphNodeNumber", line, strlen("-GraphNodeNumber"))) 
		{ 
                     sscanf(line, "-GraphNodeNumber: %d", &graphnodenumber);  
                     continue;
                }   
	
		   if (!strncmp("-GraphEdgeNumber", line, strlen("-GraphEdgeNumber")))  
               { 
                   sscanf(line, "-GraphEdgeNumber: %d", &graphedgenumber);                                  
		  
                }
	}
	fclose(fp);
	
}
void InputParameter::PrintInputParameter() {
	cout << endl << "====================" << endl << "DESIGN SPECIFICATION" << endl << "====================" << endl;
	cout << "Design Target: ";
}
