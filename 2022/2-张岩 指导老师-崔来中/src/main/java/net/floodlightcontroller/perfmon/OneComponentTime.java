package net.floodlightcontroller.perfmon;
import com.fasterxml.jackson.annotation.JsonProperty;
import net.floodlightcontroller.core.IOFMessageListener;
public class OneComponentTime {
    private String compName;
    private int pktCnt;
    private long totalProcTimeNs;
    private long maxProcTimeNs;
    private long minProcTimeNs;
    private long avgProcTimeNs;
    public OneComponentTime(IOFMessageListener module) {
        compId = module.hashCode();
        compName = module.getClass().getCanonicalName();
        resetAllCounters();
    }
    public void resetAllCounters() {
        maxProcTimeNs = Long.MIN_VALUE;
        minProcTimeNs = Long.MAX_VALUE;
        pktCnt = 0;
        totalProcTimeNs = 0;
        sumSquaredProcTimeNs2 = 0;
        avgProcTimeNs = 0;
        sigmaProcTimeNs = 0;
    }
    @JsonProperty("module-name")
    public String getCompName() {
        return compName;
    }
    @JsonProperty("num-packets")
    public int getPktCnt() {
        return pktCnt;
    }
    @JsonProperty("total")
    public long getSumProcTimeNs() {
        return totalProcTimeNs;
    }
    @JsonProperty("max")
    public long getMaxProcTimeNs() {
        return maxProcTimeNs;
    }
    @JsonProperty("min")
    public long getMinProcTimeNs() {
        return minProcTimeNs;
    }
    @JsonProperty("average")
    public long getAvgProcTimeNs() {
        return avgProcTimeNs;
    }
    @JsonProperty("std-dev")
    public long getSigmaProcTimeNs() {
        return sigmaProcTimeNs;
    }
    @JsonProperty("average-squared")
    public long getSumSquaredProcTimeNs() {
        return sumSquaredProcTimeNs2;
    }
    private void increasePktCount() {
        pktCnt++;
    }
    private void updateTotalProcessingTime(long procTimeNs) {
        totalProcTimeNs += procTimeNs;
    }
    private void updateAvgProcessTime() {
        avgProcTimeNs = totalProcTimeNs / pktCnt;
    }
    private void updateSquaredProcessingTime(long procTimeNs) {
        sumSquaredProcTimeNs2 += (Math.pow(procTimeNs, 2));
    }
    private void calculateMinProcTime(long curTimeNs) {
        if (curTimeNs < minProcTimeNs)
            minProcTimeNs = curTimeNs;
    }
    private void calculateMaxProcTime(long curTimeNs) {
        if (curTimeNs > maxProcTimeNs)
            maxProcTimeNs = curTimeNs;
    }
    public void computeSigma() {
        double temp = totalProcTimeNs;
        temp = Math.pow(temp, 2) / pktCnt;
        temp = (sumSquaredProcTimeNs2 - temp) / pktCnt;
        sigmaProcTimeNs = (long) Math.sqrt(temp);
    }
    public void updatePerPacketCounters(long procTimeNs) {
        increasePktCount();
        updateTotalProcessingTime(procTimeNs);
        calculateMinProcTime(procTimeNs);
        calculateMaxProcTime(procTimeNs);
        updateAvgProcessTime();
        updateSquaredProcessingTime(procTimeNs);
    }
    @Override
    public int hashCode() {
        return compId;
    }
}