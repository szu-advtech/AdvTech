package net.floodlightcontroller.storage.web;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.routing.Router;
import net.floodlightcontroller.restserver.RestletRoutable;
public class StorageWebRoutable implements RestletRoutable {
    @Override
    public String basePath() {
        return "/wm/storage";
    }
    @Override
    public Restlet getRestlet(Context context) {
        Router router = new Router(context);
        router.attach("/notify/json", StorageNotifyResource.class);
        return router;
    }
}
