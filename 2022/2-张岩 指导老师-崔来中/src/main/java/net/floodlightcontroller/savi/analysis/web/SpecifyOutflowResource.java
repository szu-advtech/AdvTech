package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.restlet.resource.Get;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class SpecifyOutflowResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(SpecifyOutflowResource.class);
	@Post
	@Put
	public Object update(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService == null) {
			log.error("SpecifyOutflowResource line:25 , Could not get AnalysisService!");
			return Collections.singletonMap("SpecifyOutflowResource ERROR", "Could not get analysisService!");
		}
		analysisService.updateOutFlow();
		return Collections.singletonMap("SpecifyOutflowResource", "Update Success");
	}
	@Get("json")
    public Object getOutFlow() {
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService == null) {
			log.error("SpecifyOutflowResource line:38 , Could not get AnalysisService!");
			return Collections.singletonMap("SpecifyOutflowResource ERROR", "Could not get analysisService!");
		}
		return Collections.singletonMap("SpecifyOutflowResource Result", analysisService.showOutFlow());
	}
}
