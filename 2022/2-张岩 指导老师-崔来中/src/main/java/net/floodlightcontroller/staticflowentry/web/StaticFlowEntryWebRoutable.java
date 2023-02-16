package net.floodlightcontroller.staticflowentry.web;
import net.floodlightcontroller.restserver.RestletRoutable;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.routing.Router;
public class StaticFlowEntryWebRoutable implements RestletRoutable {
    @Override
    public Restlet getRestlet(Context context) {
        Router router = new Router(context);
        router.attach("/json", StaticFlowEntryPusherResource.class);
        router.attach("/json/store", StaticFlowEntryPusherResource.class);
        router.attach("/json/delete", StaticFlowEntryDeleteResource.class);
        router.attach("/clear/{switch}/json", ClearStaticFlowEntriesResource.class);
        router.attach("/list/{switch}/json", ListStaticFlowEntriesResource.class);
        return router;
    }
    @Override
    public String basePath() {
        return "/wm/staticflowpusher";
    }
}
