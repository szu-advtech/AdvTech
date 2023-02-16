package net.floodlightcontroller.debugcounter;
public interface IDebugCounter {
    void increment();
    void add(long incr);
    long getCounterValue();
    long getLastModified();
	void reset();
}
