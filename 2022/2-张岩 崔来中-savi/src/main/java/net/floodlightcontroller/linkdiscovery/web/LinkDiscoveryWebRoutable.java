package net.floodlightcontroller.linkdiscovery.web;
import net.floodlightcontroller.restserver.RestletRoutable;
import org.restlet.Context;
import org.restlet.routing.Router;
public class LinkDiscoveryWebRoutable implements RestletRoutable {
    @Override
    public Router getRestlet(Context context) {
        Router router = new Router(context);
        return router;
    }
    @Override
    public String basePath() {
        return "/wm/linkdiscovery";
    }
}