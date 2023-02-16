package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.projectfloodlight.openflow.types.DatapathId;
public class DPIDSerializer extends JsonSerializer<DatapathId> {
    @Override
    public void serialize(DatapathId dpid, JsonGenerator jGen,
                          SerializerProvider serializer)
                                  throws IOException, JsonProcessingException {
        jGen.writeString(dpid.toString());
    }
}
