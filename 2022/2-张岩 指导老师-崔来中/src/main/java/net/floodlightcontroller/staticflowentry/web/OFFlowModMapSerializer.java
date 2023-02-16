package net.floodlightcontroller.staticflowentry.web;
import java.io.IOException;
import java.util.Map;
import net.floodlightcontroller.core.web.serializers.OFFlowModSerializer;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.JsonGenerator.Feature;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class OFFlowModMapSerializer extends JsonSerializer<OFFlowModMap> {
	@Override
	public void serialize(OFFlowModMap fmm, JsonGenerator jGen, SerializerProvider serializer)
			throws IOException, JsonProcessingException {
		if (fmm == null) {
			jGen.writeStartObject();
			jGen.writeString("No flows have been added to the Static Flow Pusher.");
			jGen.writeEndObject();
			return;
		}
		Map<String, Map<String, OFFlowMod>> theMap = fmm.getMap();
		jGen.writeStartObject();
		if (theMap.keySet() != null) {
			for (String dpid : theMap.keySet()) {
				if (theMap.get(dpid) != null) {
					jGen.writeArrayFieldStart(dpid);
					for (String name : theMap.get(dpid).keySet()) {
						jGen.writeStartObject();
						jGen.writeFieldName(name);
						OFFlowModSerializer.serializeFlowMod(jGen, theMap.get(dpid).get(name));
						jGen.writeEndObject();
					}    
					jGen.writeEndArray();
				}
			}
		}
		jGen.writeEndObject();
	}
}
