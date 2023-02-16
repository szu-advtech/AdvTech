package net.floodlightcontroller.loadbalancer;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class LBPoolSerializer extends JsonSerializer<LBPool>{
    @Override
    public void serialize(LBPool pool, JsonGenerator jGen,
                          SerializerProvider serializer) throws IOException,
                                                  JsonProcessingException {
        jGen.writeStartObject();
        jGen.writeStringField("name", pool.name);
        jGen.writeStringField("id", pool.id);
        jGen.writeStringField("vipId", pool.vipId);
        for (int i=0; i<pool.members.size(); i++)
            jGen.writeStringField("pool", pool.members.get(i));
        jGen.writeEndObject();
    }
}
