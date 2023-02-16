package net.floodlightcontroller.debugevent;
import net.floodlightcontroller.debugevent.EventResource.EventResourceBuilder;
public interface CustomFormatter<T> {
    public abstract EventResourceBuilder
            customFormat(T obj, String name, EventResourceBuilder edb);
}
