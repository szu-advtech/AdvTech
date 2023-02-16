#ifndef BANK_H_
#define BANK_H_

#include "SubArray.h"
#include "Controller.h"
class Bank  {
public:
	Bank();
	virtual ~Bank();

	/* Functions */
//	virtual void CalculateArea() = 0;
//	virtual void CalculateRC() = 0;
//	virtual void CalculateLatencyAndPower() = 0;
	virtual Bank & operator=(const Bank &);
	int bank_num;
	int master_bank_num;
	int num_subarray;                  // number of matrix in a bank
	int  capacity; 	             // The capacity of this bank
	int eDram_size;		     // The eDram_size of this bank
	int mode;                  //0 is master bank, 1 is graph bank
	Controller *controller;
	SubArray *subarray[50];
};

#endif /* BANK_H_ */
