package org.sdnplatform.sync.client;
import java.io.IOException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.MappingJsonFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
public abstract class ShellCommand {
    protected static final ObjectMapper mapper = new ObjectMapper();
    protected static final MappingJsonFactory mjf = 
            new MappingJsonFactory(mapper);
    static {
        mapper.configure(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS,
                         true);
    }
    public abstract boolean execute(String[] tokens, 
                                    String line) throws Exception;
    public abstract String syntaxString();
    protected JsonNode validateJson(JsonParser jp) throws IOException {
        JsonNode parsed = null;
        try {
            parsed = jp.readValueAsTree();
        } catch (JsonProcessingException e) {
            System.err.println("Could not parse JSON: " + e.getMessage());
            return null;
        }  
        return parsed;
    }
    protected byte[] serializeJson(JsonNode value) throws Exception {
        return mapper.writeValueAsBytes(value);
    }
}
