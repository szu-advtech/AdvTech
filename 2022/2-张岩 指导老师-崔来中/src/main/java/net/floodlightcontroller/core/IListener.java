package net.floodlightcontroller.core;
public interface IListener<T> {
    public enum Command {
        CONTINUE, STOP
    }
    public String getName();
    public boolean isCallbackOrderingPrereq(T type, String name);
    public boolean isCallbackOrderingPostreq(T type, String name);
}
