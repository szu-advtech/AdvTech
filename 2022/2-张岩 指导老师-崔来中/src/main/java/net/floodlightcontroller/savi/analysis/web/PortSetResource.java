package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class PortSetResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(PortSetResource.class);
	@Get("json")
	public Object retrieve(){
		IAnalysisService analysisService = (IAnalysisService)getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService == null) {
			log.error("PortSetResource line:19 , Could not get AnalysisService!");
			return Collections.singletonMap("PortSetResource ERROR", "Could not get AnalysisService!");
		}
		return analysisService.getPortSet();
	}
}
