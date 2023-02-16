package net.floodlightcontroller.firewall;
import net.floodlightcontroller.restserver.RestletRoutable;
import org.restlet.Context;
import org.restlet.routing.Router;
public class FirewallWebRoutable implements RestletRoutable {
    @Override
    public Router getRestlet(Context context) {
        Router router = new Router(context);
        router.attach("/module/status/json",       FirewallStatusResource.class);
        router.attach("/module/enable/json",       FirewallEnableResource.class);
        router.attach("/module/disable/json",      FirewallDisableResource.class);
        router.attach("/module/subnet-mask/json",  FirewallSubnetMaskResource.class);
        router.attach("/module/storageRules/json", FirewallStorageRulesResource.class);
        router.attach("/rules/json",               FirewallRulesResource.class);
        return router;
    }
    @Override
    public String basePath() {
        return "/wm/firewall";
    }
}
