package net.floodlightcontroller.savi.rest;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.routing.Router;
import net.floodlightcontroller.restserver.RestletRoutable;
public class SAVIRestRoute implements RestletRoutable{
	protected static final String CONVERT_TO_STATIC="static";
	protected static final String CONVERT_TO_DYNAMIC="dynamic";
	protected static final String DPID_STR = "dpid";
	@Override
	public Restlet getRestlet(Context context) {
        Router router = new Router(context);
		router.attach("/config", SAVIRest.class);
		router.attach("/flow",FlowResource.class);
		router.attach("/change/static/{"+DPID_STR+"}",ChangeTableResource.class);
		router.attach("/change/dynamic/{"+DPID_STR+"}",ChangeTableResource.class);
		return router;
	}
	@Override
	public String basePath() {
		return "/savi";
	}
}
