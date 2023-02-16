package net.floodlightcontroller.savi.analysis;
public class PacketOfFlow {
	long passNum;
	long dropNum;
	long accumulatePassNum;
	long accumulateDropNum;
	double dropRate;
	double accumulateDropRate;
	public PacketOfFlow(long passNum, long dropNum, long accumulatePassNum, long accumulateDropNum, double dropRate,
			double accumulateDropRate) {
		super();
		this.passNum = passNum;
		this.dropNum = dropNum;
		this.accumulatePassNum = accumulatePassNum;
		this.accumulateDropNum = accumulateDropNum;
		this.dropRate = dropRate;
		this.accumulateDropRate = accumulateDropRate;
	}
	public long getPassNum() {
		return passNum;
	}
	public void setPassNum(long passNum) {
		this.passNum = passNum;
	}
	public long getDropNum() {
		return dropNum;
	}
	public void setDropNum(long dropNum) {
		this.dropNum = dropNum;
	}
	public long getAccumulatePassNum() {
		return accumulatePassNum;
	}
	public void setAccumulatePassNum(long accumulatePassNum) {
		this.accumulatePassNum = accumulatePassNum;
	}
	public long getAccumulateDropNum() {
		return accumulateDropNum;
	}
	public void setAccumulateDropNum(long accumulateDropNum) {
		this.accumulateDropNum = accumulateDropNum;
	}
	public double getDropRate() {
		return dropRate;
	}
	public void setDropRate(double dropRate) {
		this.dropRate = dropRate;
	}
	public double getAccumulateDropRate() {
		return accumulateDropRate;
	}
	public void setAccumulateDropRate(double accumulateDropRate) {
		this.accumulateDropRate = accumulateDropRate;
	}
	public void init() {
		this.passNum = 0;
		this.dropNum = 0;
		this.accumulatePassNum = 0;
		this.accumulateDropNum = 0;
		this.dropRate = 0;
		this.accumulateDropRate = 0;
	}
}
