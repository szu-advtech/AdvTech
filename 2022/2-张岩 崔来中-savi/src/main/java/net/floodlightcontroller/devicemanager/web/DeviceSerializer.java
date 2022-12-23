package net.floodlightcontroller.devicemanager.web;
import java.io.IOException;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.VlanVid;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.devicemanager.internal.Device;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
public class DeviceSerializer extends JsonSerializer<Device> {
    @Override
    public void serialize(Device device, JsonGenerator jGen,
                          SerializerProvider serializer) throws IOException,
            JsonProcessingException {
        jGen.writeStartObject();
        jGen.writeStringField("entityClass", device.getEntityClass().getName());
        jGen.writeArrayFieldStart("mac");
        jGen.writeString(device.getMACAddress().toString());
        jGen.writeEndArray();
        jGen.writeArrayFieldStart("ipv4");
        for (IPv4Address ip : device.getIPv4Addresses())
            jGen.writeString(ip.toString());
        jGen.writeEndArray();
        jGen.writeArrayFieldStart("ipv6");
        for (IPv6Address ip : device.getIPv6Addresses())
            jGen.writeString(ip.toString());
        jGen.writeEndArray();
        jGen.writeArrayFieldStart("vlan");
        for (VlanVid vlan : device.getVlanId())
            if (vlan.getVlan() >= 0)
                jGen.writeString(vlan.toString());
        jGen.writeEndArray();
        jGen.writeArrayFieldStart("attachmentPoint");
        for (SwitchPort ap : device.getAttachmentPoints(true)) {
            serializer.defaultSerializeValue(ap, jGen);
        }
        jGen.writeEndArray();
        jGen.writeNumberField("lastSeen", device.getLastSeen().getTime());
        String dhcpClientName = device.getDHCPClientName();
        if (dhcpClientName != null) {
            jGen.writeStringField("dhcpClientName", dhcpClientName);
        }
        jGen.writeEndObject();
    }
}
