package net.floodlightcontroller.core.web;
import java.util.Set;
import java.util.HashSet;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.IOFSwitch;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import net.floodlightcontroller.core.web.serializers.DPIDSerializer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
public class ControllerSwitchesResource extends ServerResource {
    public static final String DPID_ERROR = "Invalid switch DPID string. Must be a 64-bit value in the form 00:11:22:33:44:55:66:77.";
    public static class DatapathIDJsonSerializerWrapper {
        private final DatapathId dpid;
        private final String inetAddress; 
        private final long connectedSince;
        public DatapathIDJsonSerializerWrapper(DatapathId dpid, String inetAddress, long connectedSince) {
            this.dpid = dpid;
            this.inetAddress = inetAddress;
            this.connectedSince = connectedSince;
        }
        @JsonSerialize(using=DPIDSerializer.class)
        public DatapathId getSwitchDPID() {
            return dpid;
        }
        public String getInetAddress() {
            return inetAddress;
        }
        public long getConnectedSince() {
            return connectedSince;
        }
    }
    @Get("json")
    public Set<DatapathIDJsonSerializerWrapper> retrieve(){
        IOFSwitchService switchService = 
            (IOFSwitchService) getContext().getAttributes().
                get(IOFSwitchService.class.getCanonicalName());
        Set<DatapathIDJsonSerializerWrapper> dpidSets = new HashSet<DatapathIDJsonSerializerWrapper>();
        for(IOFSwitch sw: switchService.getAllSwitchMap().values()) {
            dpidSets.add(new DatapathIDJsonSerializerWrapper(sw.getId(), sw.getInetAddress().toString(),  sw.getConnectedSince().getTime()));
        }
        return dpidSets;
    }
}
