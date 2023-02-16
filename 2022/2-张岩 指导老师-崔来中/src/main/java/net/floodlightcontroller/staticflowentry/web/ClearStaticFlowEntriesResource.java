package net.floodlightcontroller.staticflowentry.web;
import net.floodlightcontroller.core.web.ControllerSwitchesResource;
import net.floodlightcontroller.staticflowentry.IStaticFlowEntryPusherService;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.data.Status;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class ClearStaticFlowEntriesResource extends ServerResource {
    protected static Logger log = LoggerFactory.getLogger(ClearStaticFlowEntriesResource.class);
    @Get("json")
    public String ClearStaticFlowEntries() {
        IStaticFlowEntryPusherService sfpService =
                (IStaticFlowEntryPusherService)getContext().getAttributes().
                    get(IStaticFlowEntryPusherService.class.getCanonicalName());
        String param = (String) getRequestAttributes().get("switch");
        if (log.isDebugEnabled())
            log.debug("Clearing all static flow entires for switch: " + param);
        if (param.toLowerCase().equals("all")) {
            sfpService.deleteAllFlows();
            return "{\"status\":\"Deleted all flows.\"}";
        } else {
            try {
                sfpService.deleteFlowsForSwitch(DatapathId.of(param));
                return "{\"status\":\"Deleted all flows for switch " + param + ".\"}";
            } catch (NumberFormatException e){
                setStatus(Status.CLIENT_ERROR_BAD_REQUEST, 
                          ControllerSwitchesResource.DPID_ERROR);
                return "'{\"status\":\"Could not delete flows requested! See controller log for details.\"}'";
            }
        }
    }
}
