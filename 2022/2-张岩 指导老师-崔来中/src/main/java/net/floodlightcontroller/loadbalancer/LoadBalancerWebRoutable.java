package net.floodlightcontroller.loadbalancer;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.routing.Router;
import net.floodlightcontroller.restserver.RestletRoutable;
import net.floodlightcontroller.virtualnetwork.NoOp;
public class LoadBalancerWebRoutable implements RestletRoutable {
    @Override
    public Restlet getRestlet(Context context) {
        Router router = new Router(context);
        router.attachDefault(NoOp.class);
        return router;
     }
    @Override
    public String basePath() {
        return "/quantum/v1.0";
    }
}
