package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import org.projectfloodlight.openflow.types.MacAddress;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class MacSerializer extends JsonSerializer<MacAddress> {
    @Override
    public void serialize(MacAddress mac, JsonGenerator jGen,
                          SerializerProvider serializer)
                                  throws IOException, JsonProcessingException {
        jGen.writeString(mac.toString());
    }
}
