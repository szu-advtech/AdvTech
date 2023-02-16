package net.floodlightcontroller.virtualnetwork;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map.Entry;
import org.projectfloodlight.openflow.types.MacAddress;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class VirtualNetworkSerializer extends JsonSerializer<VirtualNetwork> {
    @Override
    public void serialize(VirtualNetwork vNet, JsonGenerator jGen,
            SerializerProvider serializer) throws IOException,
            JsonProcessingException {
        jGen.writeStartObject();
        jGen.writeStringField("name", vNet.name);
        jGen.writeStringField("guid", vNet.guid);
        jGen.writeStringField("gateway", vNet.gateway);
        jGen.writeArrayFieldStart("portMac");
		Iterator<Entry<String, MacAddress>> entries = vNet.portToMac.entrySet().iterator();
		while (entries.hasNext()){
			jGen.writeStartObject();
			Entry<String, MacAddress> entry = entries.next();
			jGen.writeStringField("port",entry.getKey().toString());
			jGen.writeStringField("mac",entry.getValue().toString());
			jGen.writeEndObject();
		}
        jGen.writeEndArray();
        jGen.writeEndObject();
    }
}
