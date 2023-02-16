package net.floodlightcontroller.core.web.serializers;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.JsonGenerator.Feature;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.ver11.OFFlowModFlagsSerializerVer11;
import org.projectfloodlight.openflow.protocol.ver12.OFFlowModFlagsSerializerVer12;
import org.projectfloodlight.openflow.protocol.ver13.OFFlowModFlagsSerializerVer13;
import org.projectfloodlight.openflow.protocol.ver14.OFFlowModFlagsSerializerVer14;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class OFFlowModSerializer extends JsonSerializer<OFFlowMod> {
    protected static Logger logger = LoggerFactory.getLogger(OFFlowModSerializer.class);
	@Override
	public void serialize(OFFlowMod fm, JsonGenerator jGen, SerializerProvider serializer)
			throws IOException, JsonProcessingException {
	}
	public static void serializeFlowMod(JsonGenerator jGen, OFFlowMod flowMod) throws IOException, JsonProcessingException {
		jGen.writeStartObject();
		jGen.writeStringField("command", flowMod.getCommand().toString());
		jGen.writeNumberField("cookie", flowMod.getCookie().getValue());
		jGen.writeNumberField("priority", flowMod.getPriority());
		jGen.writeNumberField("idleTimeoutSec", flowMod.getIdleTimeout());
		jGen.writeNumberField("hardTimeoutSec", flowMod.getHardTimeout());
		jGen.writeStringField("outPort", flowMod.getOutPort().toString());
		switch (flowMod.getVersion()) {
		case OF_10:
			break;
		case OF_11:
			jGen.writeNumberField("flags", OFFlowModFlagsSerializerVer11.toWireValue(flowMod.getFlags()));
			jGen.writeNumberField("cookieMask", flowMod.getCookieMask().getValue());
			jGen.writeStringField("outGroup", flowMod.getOutGroup().toString());
			jGen.writeStringField("tableId", flowMod.getTableId().toString());
			break;
		case OF_12:
			jGen.writeNumberField("flags", OFFlowModFlagsSerializerVer12.toWireValue(flowMod.getFlags()));
			jGen.writeNumberField("cookieMask", flowMod.getCookieMask().getValue());
			jGen.writeStringField("outGroup", flowMod.getOutGroup().toString());
			jGen.writeStringField("tableId", flowMod.getTableId().toString());
			break;
		case OF_13:
			jGen.writeNumberField("flags", OFFlowModFlagsSerializerVer13.toWireValue(flowMod.getFlags()));
			jGen.writeNumberField("cookieMask", flowMod.getCookieMask().getValue());
			jGen.writeStringField("outGroup", flowMod.getOutGroup().toString());
			break;
		case OF_14:
			jGen.writeNumberField("flags", OFFlowModFlagsSerializerVer14.toWireValue(flowMod.getFlags()));
			jGen.writeNumberField("cookieMask", flowMod.getCookieMask().getValue());
			jGen.writeStringField("outGroup", flowMod.getOutGroup().toString());
			jGen.writeStringField("tableId", flowMod.getTableId().toString());
			break;
		default:
			logger.error("Could not decode OFVersion {}", flowMod.getVersion());
			break;
		}
		MatchSerializer.serializeMatch(jGen, flowMod.getMatch());
		if (flowMod.getVersion() == OFVersion.OF_10) {
			jGen.writeObjectFieldStart("actions");
			OFActionListSerializer.serializeActions(jGen, flowMod.getActions());
			jGen.writeEndObject();
		} else {
			OFInstructionListSerializer.serializeInstructionList(jGen, flowMod.getInstructions());
		jGen.writeEndObject();
}
