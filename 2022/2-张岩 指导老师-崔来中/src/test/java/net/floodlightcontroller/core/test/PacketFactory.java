package net.floodlightcontroller.core.test;
import java.util.ArrayList;
import java.util.List;
import net.floodlightcontroller.core.IOFSwitch;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPacketInReason;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.packet.DHCP;
import net.floodlightcontroller.packet.DHCPOption;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.UDP;
public class PacketFactory {
    public static String broadcastMac = "ff:ff:ff:ff:ff:ff";
    public static String broadcastIp = "255.255.255.255";
    public static OFPacketIn DhcpDiscoveryRequestOFPacketIn(IOFSwitch sw,
            MacAddress hostMac) {
        byte[] serializedPacket = DhcpDiscoveryRequestEthernet(hostMac).serialize();
        OFFactory factory = sw.getOFFactory();
        OFPacketIn.Builder packetInBuilder = factory.buildPacketIn();
        if (factory.getVersion() == OFVersion.OF_10) {
        	packetInBuilder
        		.setInPort(OFPort.of(1))
                .setData(serializedPacket)
                .setReason(OFPacketInReason.NO_MATCH);
        } else {
        	packetInBuilder
        	.setMatch(factory.buildMatch().setExact(MatchField.IN_PORT, OFPort.of(1)).build())
            .setData(serializedPacket)
            .setReason(OFPacketInReason.NO_MATCH);
        }
        return packetInBuilder.build();
    }
    public static Ethernet DhcpDiscoveryRequestEthernet(MacAddress hostMac) {
        List<DHCPOption> optionList = new ArrayList<DHCPOption>();
        byte[] requestValue = new byte[4];
        requestValue[0] = requestValue[1] = requestValue[2] = requestValue[3] = 0;
        DHCPOption requestOption =
                new DHCPOption()
                    .setCode(DHCP.DHCPOptionCode.OptionCode_RequestedIP.
                             getValue())
                    .setLength((byte)4)
                    .setData(requestValue);
        byte[] msgTypeValue = new byte[1];
        DHCPOption msgTypeOption =
                new DHCPOption()
                    .setCode(DHCP.DHCPOptionCode.OptionCode_MessageType.
                             getValue())
                    .setLength((byte)1)
                    .setData(msgTypeValue);
        byte[] reqParamValue = new byte[4];
        DHCPOption reqParamOption =
                new DHCPOption()
                    .setCode(DHCP.DHCPOptionCode.OptionCode_RequestedParameters.
                             getValue())
                    .setLength((byte)4)
                    .setData(reqParamValue);
        byte[] clientIdValue = new byte[7];
        System.arraycopy(hostMac.getBytes(), 0,
                         clientIdValue, 1, 6);
        DHCPOption clientIdOption =
                new DHCPOption()
                    .setCode(DHCP.DHCPOptionCode.OptionCode_ClientID.
                             getValue())
                             .setLength((byte)7)
                             .setData(clientIdValue);
        DHCPOption endOption =
                new DHCPOption()
                    .setCode(DHCP.DHCPOptionCode.OptionCode_END.
                             getValue())
                             .setLength((byte)0)
                             .setData(null);
        optionList.add(requestOption);
        optionList.add(msgTypeOption);
        optionList.add(reqParamOption);
        optionList.add(clientIdOption);
        optionList.add(endOption);
        Ethernet requestPacket = new Ethernet();
        requestPacket.setSourceMACAddress(hostMac.getBytes())
        .setDestinationMACAddress(broadcastMac)
        .setEtherType(EthType.IPv4)
        .setPayload(
                new IPv4()
                .setVersion((byte)4)
                .setDiffServ((byte)0)
                .setIdentification((short)100)
                .setFlags((byte)0)
                .setFragmentOffset((short)0)
                .setTtl((byte)250)
                .setProtocol(IpProtocol.UDP)
                .setChecksum((short)0)
                .setSourceAddress(0)
                .setDestinationAddress(broadcastIp)
                .setPayload(
                        new UDP()
                        .setSourcePort(UDP.DHCP_CLIENT_PORT)
                        .setDestinationPort(UDP.DHCP_SERVER_PORT)
                        .setChecksum((short)0)
                        .setPayload(
                                new DHCP()
                                .setOpCode(DHCP.OPCODE_REQUEST)
                                .setHardwareType(DHCP.HWTYPE_ETHERNET)
                                .setHardwareAddressLength((byte)6)
                                .setHops((byte)0)
                                .setTransactionId(0x00003d1d)
                                .setSeconds((short)0)
                                .setFlags((short)0)
                                .setClientIPAddress(IPv4Address.NONE)
                                .setYourIPAddress(IPv4Address.NONE)
                                .setServerIPAddress(IPv4Address.NONE)
                                .setGatewayIPAddress(IPv4Address.NONE)
                                .setClientHardwareAddress(hostMac)
                                .setOptions(optionList))));
        return requestPacket;
    }
}
