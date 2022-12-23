package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.DataAnalysis;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class PlanResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(PlanResource.class);
	@Post
	@Put
	public Object config(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		if(analysisService == null) {
			log.error("PlanResource line:24 , Could not get AnalysisService!");
		}
		String plan = (String)getRequestAttributes().get(AnalysisWebRoutable.PLAN_TYPE);
		if(plan != null){
			int type = Integer.parseInt(plan);
			if(type != DataAnalysis.PLAN_LOSSRATE &&
					type != DataAnalysis.PLAN_TRAFFIC){
				return Collections.singletonMap("PlanChange Fail", "Invalid RequestParameters");
			}
			else {
				analysisService.changePlanByRest(type);
				return Collections.singletonMap("PlanChange Success", "Change To " + (type == DataAnalysis.PLAN_LOSSRATE ? "DropRate" : "Traffic"));
			}
		}
		return Collections.singletonMap("PlanResource ERROR", "Unimplemented configuration option");
	}
}
