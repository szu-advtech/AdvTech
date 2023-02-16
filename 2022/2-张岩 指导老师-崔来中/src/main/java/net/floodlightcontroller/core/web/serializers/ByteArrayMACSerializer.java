package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.projectfloodlight.openflow.types.MacAddress;
public class ByteArrayMACSerializer extends JsonSerializer<byte[]> {
    @Override
    public void serialize(byte[] mac, JsonGenerator jGen,
                          SerializerProvider serializer)
                                  throws IOException, JsonProcessingException {
        jGen.writeString(MacAddress.of(mac).toString());
    }
}
