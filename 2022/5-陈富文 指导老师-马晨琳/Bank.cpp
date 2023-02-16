#include "Bank.h"

Bank::Bank()
{
	// TODO Auto-generated constructor stub
}

Bank::~Bank()
{
	// TODO Auto-generated destructor stub
}

Bank &Bank::operator=(const Bank &rhs)
{
	num_subarray = rhs.num_subarray;
	// subarray= rhs.subarray;
	capacity = rhs.capacity;
	eDram_size = rhs.eDram_size;
	// miss WDD and SA and Controller
	return *this;
}
