package net.floodlightcontroller.firewall;
import org.restlet.resource.ServerResource;
class FirewallResourceBase extends ServerResource {
    IFirewallService getFirewallService() {
	return (IFirewallService)getContext().getAttributes().
	        get(IFirewallService.class.getCanonicalName());
    }
}
