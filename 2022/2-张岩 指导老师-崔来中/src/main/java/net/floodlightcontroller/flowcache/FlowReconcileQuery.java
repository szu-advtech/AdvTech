package net.floodlightcontroller.flowcache;
import java.lang.ref.SoftReference;
import net.floodlightcontroller.debugevent.IDebugEventService.EventColumn;
import net.floodlightcontroller.debugevent.IDebugEventService.EventFieldType;
import net.floodlightcontroller.debugevent.IDebugEventService.EventType;
import net.floodlightcontroller.debugevent.IEventCategory;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.debugevent.MockDebugEventService;
import net.floodlightcontroller.flowcache.PriorityPendingQueue.EventPriority;
@Deprecated
public class FlowReconcileQuery {
    public ReconcileQueryEvType evType;
    public EventPriority evPriority;
    public static class FlowReconcileQueryDebugEvent {
        @EventColumn(name = "Event Info",
                     description = EventFieldType.SREF_OBJECT)
        private final SoftReference<FlowReconcileQuery> eventInfo;
        @EventColumn(name = "Stage",
                     description = EventFieldType.STRING)
        private final String stage;
        @EventColumn(name = "Stage Info",
                     description = EventFieldType.SREF_OBJECT)
        private final SoftReference<Object> stageInfo;
        public FlowReconcileQueryDebugEvent(FlowReconcileQuery eventInfo,
                                            String stage,
                                            Object stageInfo) {
            super();
            this.eventInfo = new SoftReference<FlowReconcileQuery>(eventInfo);
            this.stage = stage;
            if (stageInfo != null) {
                this.stageInfo = new SoftReference<Object>(stageInfo);
            } else {
                this.stageInfo = null;
            }
        }
    }
    public static enum ReconcileQueryEvType {
        BVS_INTERFACE_RULE_CHANGED(EventPriority.LOW,
            "Flow Reconcile Events triggered by BVS Interface Rule Changes"),
        BVS_INTERFACE_RULE_CHANGED_MATCH_SWITCH_PORT(EventPriority.LOW,
            "Flow Reconcile Events triggered by Switch-Port based BVS Interface Rule Changes"),
        BVS_INTERFACE_RULE_CHANGED_MATCH_MAC(EventPriority.LOW,
            "Flow Reconcile Events triggered by MAC based BVS Interface Rule Changes"),
        BVS_INTERFACE_RULE_CHANGED_MATCH_VLAN(EventPriority.LOW,
            "Flow Reconcile Events triggered by VLAN based BVS Interface Rule Changes"),
        BVS_INTERFACE_RULE_CHANGED_MATCH_IPSUBNET(EventPriority.LOW,
            "Flow Reconcile Events triggered by IP Subnet based BVS Interface Rule Changes"),
        BVS_INTERFACE_RULE_CHANGED_MATCH_TAG(EventPriority.LOW,
            "Flow Reconcile Events triggered by Tag based BVS Interface Rule Changes"),
        BVS_PRIORITY_CHANGED(EventPriority.LOW,
            "Flow Reconcile Events triggered by BVS Priority Changes"),
        ACL_CONFIG_CHANGED(EventPriority.LOW,
            "Flow Reconcile Events triggered by ACL Config Changes"),
        VRS_ROUTING_RULE_CHANGED(EventPriority.LOW,
            "Flow Reconcile Events triggered by VRS Routing Rule Changes"),
        VRS_STATIC_ARP_CHANGED(EventPriority.LOW,
            "Flow Reconcile Events triggered by VRS Static ARP Config Changes"),
        DEVICE_MOVED(EventPriority.HIGH,
            "Flow Reconcile Events triggered by Host moves"),
        DEVICE_PROPERTY_CHANGED(EventPriority.LOW,
            "Flow Reconcile Events triggered by Host Property Changes"),
        LINK_DOWN(EventPriority.MEDIUM,
            "Flow Reconcile Events triggered by Link Down Events");
        private String description;
        private EventPriority priority;
        private IEventCategory<FlowReconcileQueryDebugEvent> eventCategory;
        private IDebugEventService debugEventService;
        private ReconcileQueryEvType(EventPriority priority, String description) {
            this.priority = priority;
            this.description = description;
        }
        public EventPriority getPriority() {
             return this.priority;
        }
        public String getDescription() {
            return description;
        }
        public void registerDebugEvent(String packageName, IDebugEventService debugEvents) {
        	 if (debugEventService == null) {
                 debugEventService = new MockDebugEventService();
             }
                eventCategory = debugEventService.buildEvent(FlowReconcileQueryDebugEvent.class)
                    .setModuleName(packageName)
                    .setEventName(this.toString().toLowerCase().replace("_", "-"))
                    .setEventDescription(this.getDescription())
                    .setEventType(EventType.ALWAYS_LOG)
                    .setBufferCapacity(500)
                    .register();
        }
        public IEventCategory<FlowReconcileQueryDebugEvent> getDebugEvent() {
            return eventCategory;
        }
    }
    public FlowReconcileQuery(ReconcileQueryEvType evType) {
        this.evType = evType;
        this.evPriority = evType.getPriority();
    }
    @Override
    public String toString() {
        return "FlowReconcileQuery [evType="
                + evType + "]";
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        FlowReconcileQuery other = (FlowReconcileQuery) obj;
        if (evType != other.evType) return false;
                return true;
    }
}
