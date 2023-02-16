package net.floodlightcontroller.savi.analysis.web;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class HostsCreditResource extends ServerResource {
	private static final Logger log=LoggerFactory.getLogger(HostsCreditResource.class);
	@Get("json")
	public Object retrieve() {
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		log.info("get hosts credit");
		return analysisService.getHostsCredit();
	}
}
