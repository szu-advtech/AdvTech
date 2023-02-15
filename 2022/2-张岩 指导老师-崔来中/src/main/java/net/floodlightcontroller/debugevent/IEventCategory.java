package net.floodlightcontroller.debugevent;
public interface IEventCategory<T> {
    public void newEventNoFlush(T event);
    public void newEventWithFlush(T event);
}
