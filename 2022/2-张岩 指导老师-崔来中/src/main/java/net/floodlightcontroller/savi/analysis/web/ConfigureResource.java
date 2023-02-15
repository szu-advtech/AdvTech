package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class ConfigureResource extends ServerResource {
	@Post
	@Put
	public Object config(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService==null){
			return Collections.singletonMap("ERROR", "NULL");
		}
		if (getReference().getPath().contains(AnalysisWebRoutable.ENABLE_STR)) {
			analysisService.enableAnalysis(true);
			return Collections.singletonMap("dataAnalysis", "enabled");
		}
		if (getReference().getPath().contains(AnalysisWebRoutable.DISABLE_STR)) {
			analysisService.enableAnalysis(false);
			return Collections.singletonMap("dataAnalysis", "disabled");
		}
		return Collections.singletonMap("ERROR", "Unimplemented configuration option");
	}
}
