package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import org.projectfloodlight.openflow.types.IPv4Address;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class IPv4Serializer extends JsonSerializer<IPv4Address> {
    @Override
    public void serialize(IPv4Address ipv4, JsonGenerator jGen,
                          SerializerProvider serializer)
                                  throws IOException, JsonProcessingException {
        jGen.writeString(ipv4.toString());
    }
}
