package net.floodlightcontroller.staticflowentry.web;
import java.util.Map;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
@JsonSerialize(using=OFFlowModMapSerializer.class) 
public class OFFlowModMap {
	private Map<String, Map<String, OFFlowMod>> theMap;
	public OFFlowModMap (Map<String, Map<String, OFFlowMod>> theMap) {
		this.theMap = theMap;
	}
	public Map<String, Map<String, OFFlowMod>> getMap() {
		return theMap;
	}
}
