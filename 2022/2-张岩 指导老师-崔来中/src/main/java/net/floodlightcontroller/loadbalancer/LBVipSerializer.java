package net.floodlightcontroller.loadbalancer;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class LBVipSerializer extends JsonSerializer<LBVip>{
    @Override
    public void serialize(LBVip vip, JsonGenerator jGen,
                          SerializerProvider serializer) throws IOException,
                                                  JsonProcessingException {
        jGen.writeStartObject();
        jGen.writeStringField("name", vip.name);
        jGen.writeStringField("id", vip.id);
        jGen.writeStringField("address", String.valueOf(vip.address));
        jGen.writeStringField("protocol", Byte.toString(vip.protocol));
        jGen.writeStringField("port", Short.toString(vip.port));
        jGen.writeEndObject();
    }
}
