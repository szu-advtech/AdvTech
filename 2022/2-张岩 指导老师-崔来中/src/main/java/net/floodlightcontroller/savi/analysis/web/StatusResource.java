package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.DataAnalysis;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class StatusResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(StatusResource.class);
	@Post
	@Put
	public Object config(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService == null) {
			log.error("StatusResource line:21 , Could not get AnalysisService!");
		}
		if(getReference().getPath().contains(AnalysisWebRoutable.STAGE_INIT)){
			analysisService.changeStatusByRest(DataAnalysis.INIT_STAGE);
			return Collections.singletonMap("DataAnalysis", "Init");
		}
		return Collections.singletonMap("StatusResouce ERROR", "Unimplemented configuration option");
	}
}
