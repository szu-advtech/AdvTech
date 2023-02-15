package net.floodlightcontroller.firewall;
import org.restlet.resource.Get;
import org.restlet.resource.Put;
import org.restlet.data.Status;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class FirewallDisableResource extends FirewallResourceBase {
    private static final Logger log = LoggerFactory.getLogger(FirewallDisableResource.class);
    @Get("json")
    public Object handleRequest() {
        log.warn("call to FirewallDisableResource with method GET is not allowed. Use PUT: ");
        setStatus(Status.CLIENT_ERROR_METHOD_NOT_ALLOWED);
	return "{\"status\" : \"failure\", \"details\" : \"Use PUT to disable firewall\"}";
    }
    @Put("json")
    public Object handlePut() {
        IFirewallService firewall = getFirewallService();
	firewall.enableFirewall(false);
        setStatus(Status.SUCCESS_OK);
	return "{\"status\" : \"success\", \"details\" : \"firewall stopped\"}";
    }
}
