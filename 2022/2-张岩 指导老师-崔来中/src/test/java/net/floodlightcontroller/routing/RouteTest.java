package net.floodlightcontroller.routing;
import org.junit.Test;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.routing.Route;
import net.floodlightcontroller.test.FloodlightTestCase;
import net.floodlightcontroller.topology.NodePortTuple;
public class RouteTest extends FloodlightTestCase {
    @Test
    public void testCloneable() throws Exception {
        Route r1 = new Route(DatapathId.of(1L), DatapathId.of(2L));
        Route r2 = new Route(DatapathId.of(1L), DatapathId.of(3L));
        assertNotSame(r1, r2);
        assertNotSame(r1.getId(), r2.getId());
        r1 = new Route(DatapathId.of(1L), DatapathId.of(3L));
        r1.getPath().add(new NodePortTuple(DatapathId.of(1L), OFPort.of((short)1)));
        r1.getPath().add(new NodePortTuple(DatapathId.of(2L), OFPort.of((short)1)));
        r1.getPath().add(new NodePortTuple(DatapathId.of(2L), OFPort.of((short)2)));
        r1.getPath().add(new NodePortTuple(DatapathId.of(3L), OFPort.of((short)1)));
        r2.getPath().add(new NodePortTuple(DatapathId.of(1L), OFPort.of((short)1)));
        r2.getPath().add(new NodePortTuple(DatapathId.of(2L), OFPort.of((short)1)));
        r2.getPath().add(new NodePortTuple(DatapathId.of(2L), OFPort.of((short)2)));
        r2.getPath().add(new NodePortTuple(DatapathId.of(3L), OFPort.of((short)1)));
        assertEquals(r1, r2);
        NodePortTuple temp = r2.getPath().remove(0);
        assertNotSame(r1, r2);
        r2.getPath().add(0, temp);
        assertEquals(r1, r2);
        r2.getPath().remove(1);
        temp = new NodePortTuple(DatapathId.of(2L), OFPort.of((short)5));
        r2.getPath().add(1, temp);
        assertNotSame(r1, r2);
    }
}
