package net.floodlightcontroller.topology.web;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.topology.ITopologyService;
import net.floodlightcontroller.topology.NodePortTuple;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
public class EnabledPortsResource extends ServerResource {
    @Get("json")
    public List<NodePortTuple> retrieve() {
        List<NodePortTuple> result = new ArrayList<NodePortTuple>();
        IOFSwitchService switchService =
                (IOFSwitchService) getContext().getAttributes().
                get(IOFSwitchService.class.getCanonicalName());
        ITopologyService topologyService =
                (ITopologyService) getContext().getAttributes().
                get(ITopologyService.class.getCanonicalName());
        if (switchService == null || topologyService == null)
            return result;
        Set<DatapathId> switches = switchService.getAllSwitchDpids();
        if (switches == null) return result;
        for(DatapathId sw: switches) {
            Set<OFPort> ports = topologyService.getPorts(sw);
            if (ports == null) continue;
            for(OFPort p: ports) {
                result.add(new NodePortTuple(sw, p));
            }
        }
        return result;
    }
}
