package net.floodlightcontroller.core.util;
public class MutableInteger extends Number {
    private static final long serialVersionUID = 1L;
    int mutableInt;
    public MutableInteger(int value) {
        this.mutableInt = value;
    }
    public void setValue(int value) {
        this.mutableInt = value;
    }
    @Override
    public double doubleValue() {
        return (double) mutableInt;
    }
    @Override
    public float floatValue() {
        return (float) mutableInt;
    }
    @Override
    public int intValue() {
        return mutableInt;
    }
    @Override
    public long longValue() {
        return (long) mutableInt;
    }
}
