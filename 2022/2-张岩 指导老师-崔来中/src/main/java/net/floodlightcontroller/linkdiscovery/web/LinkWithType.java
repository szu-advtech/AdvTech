package net.floodlightcontroller.linkdiscovery.web;
import java.io.IOException;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LinkDirection;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LinkType;
import net.floodlightcontroller.routing.Link;
@JsonSerialize(using=LinkWithType.class)
public class LinkWithType extends JsonSerializer<LinkWithType> {
    public DatapathId srcSwDpid;
    public OFPort srcPort;
    public DatapathId dstSwDpid;
    public OFPort dstPort;
    public LinkType type;
    public LinkDirection direction;
    public LinkWithType() {}
    public LinkWithType(Link link,
            LinkType type,
            LinkDirection direction) {
        this.srcSwDpid = link.getSrc();
        this.srcPort = link.getSrcPort();
        this.dstSwDpid = link.getDst();
        this.dstPort = link.getDstPort();
        this.type = type;
        this.direction = direction;
    }
    @Override
    public void serialize(LinkWithType lwt, JsonGenerator jgen, SerializerProvider arg2)
            throws IOException, JsonProcessingException {
        jgen.writeStartObject();
        jgen.writeStringField("src-switch", lwt.srcSwDpid.toString());
        jgen.writeNumberField("src-port", lwt.srcPort.getPortNumber());
        jgen.writeStringField("dst-switch", lwt.dstSwDpid.toString());
        jgen.writeNumberField("dst-port", lwt.dstPort.getPortNumber());
        jgen.writeStringField("type", lwt.type.toString());
        jgen.writeStringField("direction", lwt.direction.toString());
        jgen.writeEndObject();
    }
    @Override
    public Class<LinkWithType> handledType() {
        return LinkWithType.class;
    }
}