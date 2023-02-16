package net.floodlightcontroller.savi.analysis.web;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class RuleNumberResource extends ServerResource{
	Logger log=LoggerFactory.getLogger(RuleNumberResource.class);
	@Get("json")
	public Object retrieve() {
		IAnalysisService analysisService=(IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		String swStr=(String) getRequestAttributes().get(AnalysisWebRoutable.SWITCH_ID);
		if(swStr.equalsIgnoreCase("all")){
			log.info("sucess");
			return analysisService.calculateRule();
		}
		DatapathId dpid;
		try {
			dpid=DatapathId.of(swStr);
		} catch (Exception e) {
			log.error("Error, unable to parse switch id");
			return null;
		}
		return analysisService.calculateRule(dpid);
	}
}
