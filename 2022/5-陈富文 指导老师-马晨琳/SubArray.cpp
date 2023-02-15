
#include "SubArray.h"
#include <math.h>

SubArray::SubArray() {
	// TODO Auto-generated constructor stub
	bitlineDelay=100;
	chargeLatency=100;
	columnDecoderLatency=100;
}

SubArray::~SubArray() {
	// TODO Auto-generated destructor stub
}

void SubArray::Initialize(int _numRow, int _numColumn, int _muxSenseAmp)   // inital
{
	numRow=_numRow;
	numColumn=_numColumn;
	muxSenseAmp=_muxSenseAmp;
	int i, j;
	for(i=0;i<numRow;i++)
		for(j=0;j<numColumn;j++)
			cells[i][j]=0;
	empty_row=0;
	empty_column=0;
	sum_cell=0;
}
int SubArray::write_cell(int _numRow, int _numColumn, int value)  // 0 success, 1 failed
{
	if(_numRow>numRow|| _numColumn>numColumn)
		return 1;
	cells[_numRow-1][_numColumn-1]=value;
	return 0;
}
int SubArray::read_cell(int _numRow, int _numColumn)  // 0 success, -1 failed
{
        if(_numRow>numRow|| _numColumn>numColumn)
                return -1;
        return cells[_numRow-1][_numColumn-1];
}

