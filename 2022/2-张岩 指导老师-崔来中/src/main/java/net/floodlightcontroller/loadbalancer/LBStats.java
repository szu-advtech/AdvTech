package net.floodlightcontroller.loadbalancer;
public class LBStats {
    protected int bytesIn;
    protected int bytesOut;
    protected int activeConnections;
    protected int totalConnections;
    public LBStats() {
        bytesIn = 0;
        bytesOut = 0;
        activeConnections = 0;
        totalConnections = 0;
    }
}
