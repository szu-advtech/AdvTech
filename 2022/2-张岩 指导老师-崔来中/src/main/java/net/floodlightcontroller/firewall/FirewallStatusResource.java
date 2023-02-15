package net.floodlightcontroller.firewall;
import org.restlet.resource.Get;
public class FirewallStatusResource extends FirewallResourceBase {
    @Get("json")
    public Object handleRequest() {
        IFirewallService firewall = this.getFirewallService();
	if (firewall.isEnabled())
	    return "{\"result\" : \"firewall enabled\"}";
	else
	    return "{\"result\" : \"firewall disabled\"}";
    }
}
