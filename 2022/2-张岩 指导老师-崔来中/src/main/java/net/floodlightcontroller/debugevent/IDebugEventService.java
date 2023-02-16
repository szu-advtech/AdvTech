package net.floodlightcontroller.debugevent;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.List;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.debugevent.DebugEventResource.EventInfoResource;
import net.floodlightcontroller.debugevent.DebugEventService.EventCategory;
import net.floodlightcontroller.debugevent.DebugEventService.EventCategoryBuilder;
public interface IDebugEventService extends IFloodlightService {
    public enum EventType {
        ALWAYS_LOG, LOG_ON_DEMAND
    }
    public enum AckableEvent {
        ACKABLE, NOT_ACKABLE
    }
    enum EventFieldType {
        DPID, IPv4, IPv6, MAC, STRING, OBJECT, PRIMITIVE, COLLECTION_IPV4,
        COLLECTION_ATTACHMENT_POINT, COLLECTION_OBJECT, SREF_COLLECTION_OBJECT,
        SREF_OBJECT
    }
    @Target(ElementType.FIELD)
    @Retention(RetentionPolicy.RUNTIME)
    public @interface EventColumn {
        String name() default "param";
        EventFieldType description() default EventFieldType.PRIMITIVE;
    }
    public static final String EV_MDATA_WARN = "warn";
    public static final String EV_MDATA_ERROR = "error";
    public <T> EventCategoryBuilder<T> buildEvent(Class<T> evClass);
    public void flushEvents();
    public boolean containsModuleEventName(String moduleName,
                                           String eventName);
    public boolean containsModuleName(String moduleName);
    public List<EventInfoResource> getAllEventHistory();
    public List<EventInfoResource> getModuleEventHistory(String moduleName);
    public EventInfoResource getSingleEventHistory(String moduleName,
                                                   String eventName,
                                                   int numOfEvents);
    public void resetAllEvents();
    public void resetAllModuleEvents(String moduleName);
    public void resetSingleEvent(String moduleName, String eventName);
    public List<String> getModuleList();
    public List<String> getModuleEventList(String moduleName);
    public void setAck(int eventId, long eventInstanceId, boolean ack);
}
