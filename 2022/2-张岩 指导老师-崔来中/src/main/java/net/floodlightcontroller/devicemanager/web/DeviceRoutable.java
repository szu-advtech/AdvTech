package net.floodlightcontroller.devicemanager.web;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.routing.Router;
import net.floodlightcontroller.restserver.RestletRoutable;
public class DeviceRoutable implements RestletRoutable {
    @Override
    public String basePath() {
        return "/wm/device";
    }
    @Override
    public Restlet getRestlet(Context context) {
        Router router = new Router(context);
        router.attach("/all/json", DeviceResource.class);
        router.attach("/", DeviceResource.class);
        router.attach("/debug", DeviceEntityResource.class);
        return router;
    }
}
