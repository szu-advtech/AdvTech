package net.floodlightcontroller.savi.analysis.web;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class PacketOfAllFlowResource extends ServerResource {
	@Get("json")
	public Object retrieve(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		return analysisService.getAllPacketOfFlow();
	}
}
