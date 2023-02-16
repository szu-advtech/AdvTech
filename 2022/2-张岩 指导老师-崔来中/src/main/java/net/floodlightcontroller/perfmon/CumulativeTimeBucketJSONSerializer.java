package net.floodlightcontroller.perfmon;
import java.io.IOException;
import java.sql.Timestamp;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class CumulativeTimeBucketJSONSerializer
                                extends JsonSerializer<CumulativeTimeBucket> {
   @Override
   public void serialize(CumulativeTimeBucket ctb,
                   JsonGenerator jGen,
                   SerializerProvider serializer) 
                   throws IOException, JsonProcessingException {
       jGen.writeStartObject();
       Timestamp ts = new Timestamp(ctb.getStartTimeNs()/1000000);
       jGen.writeStringField("start-time", ts.toString());
       jGen.writeStringField("current-time", 
         new Timestamp(System.currentTimeMillis()).toString());
       jGen.writeNumberField("total-packets", ctb.getTotalPktCnt());
       jGen.writeNumberField("average", ctb.getAverageProcTimeNs());
       jGen.writeNumberField("min", ctb.getMinTotalProcTimeNs());
       jGen.writeNumberField("max", ctb.getMaxTotalProcTimeNs());
       jGen.writeNumberField("std-dev", ctb.getTotalSigmaProcTimeNs());
       jGen.writeArrayFieldStart("modules");
       for (OneComponentTime oct : ctb.getModules()) {
           serializer.defaultSerializeValue(oct, jGen);
       }
       jGen.writeEndArray();
       jGen.writeEndObject();
   }
   @Override
   public Class<CumulativeTimeBucket> handledType() {
       return CumulativeTimeBucket.class;
   }
}
