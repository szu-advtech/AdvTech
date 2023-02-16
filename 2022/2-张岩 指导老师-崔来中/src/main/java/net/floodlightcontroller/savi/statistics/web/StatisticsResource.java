package net.floodlightcontroller.savi.statistics.web;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import net.floodlightcontroller.savi.statistics.IStatisticsCollector;
public class StatisticsResource extends ServerResource {
	@Get("json")
	public Object getJson() {
		IStatisticsCollector collector = (IStatisticsCollector)getContext().getAttributes().get(IStatisticsCollector.class.getCanonicalName());
		return collector.getStatistics();
	}
}
