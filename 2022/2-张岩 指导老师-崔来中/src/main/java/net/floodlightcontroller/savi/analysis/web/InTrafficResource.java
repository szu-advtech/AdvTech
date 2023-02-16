package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.restlet.resource.Get;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class InTrafficResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(InTrafficResource.class);
	@Post
	@Put
	public Object update(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService == null) {
			log.error("InTrafficResource line:24 , Could not get AnalysisService!");
			return Collections.singletonMap("InTrafficResource ERROR", "Could not get analysisService!");
		}
		analysisService.updateMaxTraffic();
		return Collections.singletonMap("InTrafficResource Success", "Update Success");
	}
	@Get("json")
    public Object getOutFlow() {
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService == null) {
			log.error("InTrafficResource line:37 , Could not get AnalysisService!");
			return Collections.singletonMap("InTrafficResource ERROR", "Could not get analysisService!");
		}
		return Collections.singletonMap("InTrafficResource Result", analysisService.showMaxTraffic());
	}
}
