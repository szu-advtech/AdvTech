package net.floodlightcontroller.routing;
import java.util.ArrayList;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.routing.Route;
public interface IRoutingService extends IFloodlightService {
    public Route getRoute(DatapathId src, DatapathId dst, U64 cookie);
    public Route getRoute(DatapathId src, DatapathId dst, U64 cookie, boolean tunnelEnabled);
    public Route getRoute(DatapathId srcId, OFPort srcPort, DatapathId dstId, OFPort dstPort, U64 cookie);
    public Route getRoute(DatapathId srcId, OFPort srcPort, DatapathId dstId, OFPort dstPort, U64 cookie, boolean tunnelEnabled);
    public ArrayList<Route> getRoutes(DatapathId longSrcDpid, DatapathId longDstDpid, boolean tunnelEnabled);
    public boolean routeExists(DatapathId src, DatapathId dst);
    public boolean routeExists(DatapathId src, DatapathId dst, boolean tunnelEnabled);
}
