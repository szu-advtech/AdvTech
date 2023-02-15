#ifndef SUBARRAY_H_
#define SUBARRAY_H_

#define MAX_COLUMN 1024
class SubArray {
public:
	SubArray();
	virtual ~SubArray();

	/* Functions */
	void Initialize(int , int , int );
	int write_cell(int _numRow, int _numColumn, int value); //0 means successful, 1 means failed
	int read_cell(int _numRow, int _numColumn); // 0 success, -1 failed

	int numRow;
	int numColumn;
	int cells[1024][1024];
	int empty_row;
	int empty_column;
	int sum_cell;
	int muxSenseAmp;  // how many bitlines connect to one sense amplifier
	bool voltageSense;  //whether the sense amplifier is voltage-sensing
	double bitlineDelay;	/* Bitline delay, Unit: s */
	double chargeLatency;	/* The bitline charge delay during write operations, Unit: s */
	double columnDecoderLatency;	/* The worst-case mux latency, Unit: s */
};

#endif /* SUBARRAY_H_ */
