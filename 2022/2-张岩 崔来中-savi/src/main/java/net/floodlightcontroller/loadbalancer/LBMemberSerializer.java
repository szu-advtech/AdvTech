package net.floodlightcontroller.loadbalancer;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class LBMemberSerializer extends JsonSerializer<LBMember>{
    @Override
    public void serialize(LBMember member, JsonGenerator jGen,
                          SerializerProvider serializer) throws IOException,
                                                  JsonProcessingException {
        jGen.writeStartObject();
        jGen.writeStringField("id", member.id);
        jGen.writeStringField("address", String.valueOf(member.address));
        jGen.writeStringField("port", Short.toString(member.port));
        jGen.writeStringField("poolId", member.poolId);
        jGen.writeStringField("vipId", member.vipId);
        jGen.writeEndObject();
    }
}
