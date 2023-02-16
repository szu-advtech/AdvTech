package net.floodlightcontroller.savi.statistics.web;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.resource.Directory;
import org.restlet.routing.Router;
import net.floodlightcontroller.restserver.RestletRoutable;
public class StatisticsWebRoutable implements RestletRoutable {
	@Override
	public Restlet getRestlet(Context context) {
		Router router = new Router(context);
		router.attach("/json", StatisticsResource.class);
		return router;
	}
	@Override
	public String basePath() {
		return "/statistics";
	}
}
