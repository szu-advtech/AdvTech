package net.floodlightcontroller.savi.rest;
import java.util.Collections;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.service.SAVIProviderService;
public class ChangeTableResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(ChangeTableResource.class);
	@Post
	@Put
	public Object config(){
		SAVIProviderService saviProviderService = (SAVIProviderService) getContext().getAttributes().get(SAVIProviderService.class.getCanonicalName());
		if(saviProviderService == null) {
			log.error("StatusResource line:21 , Could not get AnalysisService!");
		}
		String d = (String) getRequestAttributes().get(SAVIRestRoute.DPID_STR);
		DatapathId dpid = DatapathId.NONE;
		if (!d.trim().equalsIgnoreCase("all")) {
			try {
				dpid = DatapathId.of(d);
			} catch (Exception e) {
				log.error("Could not parse DPID {}", d);
				return Collections.singletonMap("ERROR", "Could not parse DPID" + d);
			}
		}
		if(getReference().getPath().contains(SAVIRestRoute.CONVERT_TO_STATIC)){
			if (dpid.equals(DatapathId.NONE)) {
				saviProviderService.convertTable(true);
			}else {
				saviProviderService.convertTable(dpid,true);
			}
			return Collections.singletonMap("changeToStatic", dpid+" sucess to change to static");
		}
		if(getReference().getPath().contains(SAVIRestRoute.CONVERT_TO_DYNAMIC)){
			if (dpid.equals(DatapathId.NONE)) {
				saviProviderService.convertTable(false);
			}else {
				saviProviderService.convertTable(dpid,false);
			}
			return Collections.singletonMap("changeToDynamic", dpid+" sucess to change to dynamic");
		}
		return Collections.singletonMap("ChangeTableResource ERROR", "Unimplemented configuration option");
	}
}
