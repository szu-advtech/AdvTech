package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class UShortSerializer extends JsonSerializer<Short> {
    @Override
    public void serialize(Short s, JsonGenerator jGen,
                          SerializerProvider serializer) throws IOException,
                                                  JsonProcessingException {
        if (s == null) jGen.writeNull();
        else jGen.writeNumber(s.shortValue() & 0xffff);
    }
}
