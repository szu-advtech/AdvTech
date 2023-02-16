package net.floodlightcontroller.staticflowentry.web;
import java.util.HashMap;
import java.util.Map;
import net.floodlightcontroller.core.web.ControllerSwitchesResource;
import net.floodlightcontroller.staticflowentry.IStaticFlowEntryPusherService;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.data.Status;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class ListStaticFlowEntriesResource extends ServerResource {
    protected static Logger log = LoggerFactory.getLogger(ListStaticFlowEntriesResource.class);
    @Get("json")
    public OFFlowModMap ListStaticFlowEntries() {
        IStaticFlowEntryPusherService sfpService =
                (IStaticFlowEntryPusherService)getContext().getAttributes().
                    get(IStaticFlowEntryPusherService.class.getCanonicalName());
        String param = (String) getRequestAttributes().get("switch");
        if (log.isDebugEnabled())
            log.debug("Listing all static flow entires for switch: " + param);
        if (param.toLowerCase().equals("all")) {
        	System.out.println("11111111111111111这里return");
            return new OFFlowModMap(sfpService.getFlows());
        } else {
            try {
                Map<String, Map<String, OFFlowMod>> retMap = new HashMap<String, Map<String, OFFlowMod>>();
                retMap.put(param, sfpService.getFlows(DatapathId.of(param)));
                System.out.println("2222222222222222这里return");
                return new OFFlowModMap(retMap);
            } catch (NumberFormatException e){
                setStatus(Status.CLIENT_ERROR_BAD_REQUEST, ControllerSwitchesResource.DPID_ERROR);
            }
        }
        System.out.println("3333333333333这里return");
        return null;
    }
}
