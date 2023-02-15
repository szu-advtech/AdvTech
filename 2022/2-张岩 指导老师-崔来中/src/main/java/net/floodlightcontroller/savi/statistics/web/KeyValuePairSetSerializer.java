package net.floodlightcontroller.savi.statistics.web;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class KeyValuePairSetSerializer extends JsonSerializer<KeyValuePairSet> {
	@Override
	public void serialize(KeyValuePairSet arg0, JsonGenerator arg1, SerializerProvider arg2)
			throws IOException, JsonProcessingException {
		arg1.writeStartObject();
		for(KeyValuePair pair:arg0.getPairs()) {
			arg1.writeStringField(pair.getKey(), pair.getValue());
		}
		arg1.writeEndObject();
	}
}
