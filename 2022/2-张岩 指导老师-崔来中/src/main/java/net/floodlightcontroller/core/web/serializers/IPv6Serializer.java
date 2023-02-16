package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import org.projectfloodlight.openflow.types.IPv6Address;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class IPv6Serializer extends JsonSerializer<IPv6Address> {
    @Override
    public void serialize(IPv6Address ipv6, JsonGenerator jGen,
                          SerializerProvider serializer)
                                  throws IOException, JsonProcessingException {
        jGen.writeString(ipv6.toString());
    }
}
