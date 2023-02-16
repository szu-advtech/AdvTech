package net.floodlightcontroller.firewall;
import org.restlet.resource.Get;
public class FirewallStorageRulesResource extends FirewallResourceBase {
	@Get("json")
	public Object handleRequest() {
		IFirewallService firewall = getFirewallService();
		return firewall.getStorageRules();
	}
}
