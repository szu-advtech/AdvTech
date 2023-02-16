package net.floodlightcontroller.debugevent;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
import net.floodlightcontroller.debugevent.Event;
import net.floodlightcontroller.debugevent.EventResource;
import net.floodlightcontroller.debugevent.EventResource.EventResourceBuilder;
import net.floodlightcontroller.debugevent.EventResource.Metadata;
import net.floodlightcontroller.debugevent.IDebugEventService.EventColumn;
import net.floodlightcontroller.debugevent.IDebugEventService.EventFieldType;
import org.projectfloodlight.openflow.types.DatapathId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class EventTest {
    protected static Logger log = LoggerFactory.getLogger(EventTest.class);
    @Test
    public void testFormat() {
        River r = new River("ganges", 42);
        Event e = new Event(1L, 32, "test",
                      new RiverEvent(DatapathId.of(1L), (short)10, true, "big river", 5, 4L, r), 10L);
        EventResourceBuilder edb = new EventResourceBuilder();
        edb.dataFields.add(new Metadata("dpid", "00:00:00:00:00:00:00:01"));
        edb.dataFields.add(new Metadata("portId", "10"));
        edb.dataFields.add(new Metadata("valid", "true"));
        edb.dataFields.add(new Metadata("desc", "big river"));
        edb.dataFields.add(new Metadata("ip", "0.0.0.5"));
        edb.dataFields.add(new Metadata("mac", "00:00:00:00:00:04"));
        edb.dataFields.add(new Metadata("obj", "ganges/42"));
        edb.setThreadId(e.getThreadId());
        edb.setThreadName(e.getThreadName());
        edb.setTimeStamp(e.getTimeMs());
        edb.setModuleEventName("test");
        EventResource ed = edb.build();
        assertTrue(ed.equals(e.getFormattedEvent(RiverEvent.class, "test")));
        Pattern pat =
                Pattern.compile("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}[+-]\\d{2}:\\d{2}");
        Date t1 = e.getFormattedEvent(RiverEvent.class, "test2").getTimestamp();
        Matcher m1 = pat.matcher(t1.toString());
        assertTrue(m1.matches());
        assertFalse(ed.equals(e.getFormattedEvent(River.class, "test")));
        assertTrue(e.getFormattedEvent(River.class, "test").getDataFields().
                   contains(new Metadata("Error",
                   "null event data or event-class does not match event-data")));
        assertTrue(e.getFormattedEvent(null, "test").getDataFields().contains(
          new Metadata("Error",
                   "null event data or event-class does not match event-data")));
    }
    @Test
    public void testIncorrectAnnotation() {
        Event e = new Event(1L, 32, "test",
        assertTrue(e.getFormattedEvent(LakeEvent.class, "test").getDataFields()
          .contains(new Metadata("Error",
                             "java.lang.Integer cannot be cast to org.projectfloodlight.openflow.types.DatapathId")));
        Event e2 = new Event(1L, 32, "test",
        assertTrue(e2.getFormattedEvent(LakeEvent2.class, "test").getDataFields()
                   .contains(new Metadata("Error",
                                      "java.lang.Integer cannot be cast to java.lang.Long")));
    }
    class RiverEvent  {
        @EventColumn(name = "dpid", description = EventFieldType.DPID)
        DatapathId dpid;
        @EventColumn(name = "portId", description = EventFieldType.PRIMITIVE)
        short srcPort;
        @EventColumn(name = "valid", description = EventFieldType.PRIMITIVE)
        boolean isValid;
        @EventColumn(name = "desc", description = EventFieldType.STRING)
        String desc;
        @EventColumn(name = "ip", description = EventFieldType.IPv4)
        int ipAddr;
        @EventColumn(name = "mac", description = EventFieldType.MAC)
        long macAddr;
        @EventColumn(name = "obj", description = EventFieldType.OBJECT)
        River amazon;
        public RiverEvent(DatapathId dpid, short srcPort, boolean isValid,
                            String desc, int ip, long mac, River passedin) {
            this.dpid = dpid;
            this.srcPort = srcPort;
            this.isValid = isValid;
            this.desc = desc;
            this.ipAddr = ip;
            this.macAddr = mac;
        }
    }
    class River {
        String r1;
        long r2;
        public River(String r1, long r2) {
            this.r1 = r1;
            this.r2 = r2;
        }
        public River(River passedin) {
            this.r1 = passedin.r1;
            this.r2 = passedin.r2;
        }
        @Override
        public String toString() {
            return (r1 + "/" + r2);
        }
    }
    class LakeEvent {
        @EventColumn(name = "dpid", description = EventFieldType.DPID)
        int dpid;
        public LakeEvent(int dpid) {
            this.dpid = dpid;
        }
    }
    class LakeEvent2 {
        @EventColumn(name = "mac", description = EventFieldType.MAC)
        int mac;
        public LakeEvent2(int mac) {
            this.mac = mac;
        }
    }
}