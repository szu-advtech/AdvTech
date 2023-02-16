package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class AutoCheckResource extends ServerResource{
	IAnalysisService analysisService =(IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
	@Post
	@Put
	public Object config() {
		String autoCheck=(String) getRequestAttributes().get(AnalysisWebRoutable.ISAUTO);
		if(autoCheck.toLowerCase().equals("true")) {
			analysisService.setAutoCheck(true);
			return Collections.singletonMap("success", "change to auto check");
		}else if(autoCheck.toLowerCase().equals("false")) {
			analysisService.setAutoCheck(false);
			return Collections.singletonMap("success", "change to no check");
		}else {
			return Collections.singletonMap("ERROR", "input the wrong site");
		}
	}
}
