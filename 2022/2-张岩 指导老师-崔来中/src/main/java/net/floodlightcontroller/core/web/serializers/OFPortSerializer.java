package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.projectfloodlight.openflow.types.OFPort;
public class OFPortSerializer extends JsonSerializer<OFPort> {
    @Override
    public void serialize(OFPort port, JsonGenerator jGen,
                          SerializerProvider serializer)
                                  throws IOException, JsonProcessingException {
        jGen.writeNumber(port.getPortNumber());
    }
}
