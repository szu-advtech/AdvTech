package net.floodlightcontroller.topology.web;
import java.util.List;
import net.floodlightcontroller.routing.IRoutingService;
import net.floodlightcontroller.routing.Route;
import net.floodlightcontroller.topology.NodePortTuple;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class RouteResource extends ServerResource {
    protected static Logger log = LoggerFactory.getLogger(RouteResource.class);
    @Get("json")
    public List<NodePortTuple> retrieve() {
        IRoutingService routing = 
                (IRoutingService)getContext().getAttributes().
                    get(IRoutingService.class.getCanonicalName());
        String srcDpid = (String) getRequestAttributes().get("src-dpid");
        String srcPort = (String) getRequestAttributes().get("src-port");
        String dstDpid = (String) getRequestAttributes().get("dst-dpid");
        String dstPort = (String) getRequestAttributes().get("dst-port");
        log.debug( srcDpid + "--" + srcPort + "--" + dstDpid + "--" + dstPort);
        DatapathId longSrcDpid = DatapathId.of(srcDpid);
        OFPort shortSrcPort = OFPort.of(Integer.parseInt(srcPort));
        DatapathId longDstDpid = DatapathId.of(dstDpid);
        OFPort shortDstPort = OFPort.of(Integer.parseInt(dstPort));
        Route result = routing.getRoute(longSrcDpid, shortSrcPort, longDstDpid, shortDstPort, U64.of(0));
        if (result != null) {
            return routing.getRoute(longSrcDpid, shortSrcPort, longDstDpid, shortDstPort, U64.of(0)).getPath();
        }
        else {
            log.debug("ERROR! no route found");
            return null;
        }
    }
}
