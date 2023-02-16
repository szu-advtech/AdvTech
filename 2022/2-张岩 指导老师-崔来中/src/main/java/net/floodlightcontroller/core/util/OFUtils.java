package net.floodlightcontroller.core.util;
import java.nio.ByteBuffer;
import java.util.Objects;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.ArpOpcode;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.ICMPv4Code;
import org.projectfloodlight.openflow.types.ICMPv4Type;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IpDscp;
import org.projectfloodlight.openflow.types.IpEcn;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.OFVlanVidMatch;
import org.projectfloodlight.openflow.types.TransportPort;
import org.projectfloodlight.openflow.types.U8;
import org.projectfloodlight.openflow.types.VlanPcp;
public class OFUtils {
    public static Match loadFromPacket(byte[] packetData, OFPort inputPort,
            OFFactory factory) {
        short scratch;
        int transportOffset = 34;
        ByteBuffer packetDataBB = ByteBuffer.wrap(packetData);
        int limit = packetDataBB.limit();
        assert (limit >= 14);
        Match.Builder builder = factory.buildMatch();
        if (!Objects.equals(inputPort, OFPort.ALL))
            builder.setExact(MatchField.IN_PORT, inputPort);
        byte[] dataLayerDestination = new byte[6];
        packetDataBB.get(dataLayerDestination);
        builder.setExact(MatchField.ETH_DST, MacAddress.of(dataLayerDestination));
        byte[] dataLayerSource = new byte[6];
        packetDataBB.get(dataLayerSource);
        builder.setExact(MatchField.ETH_SRC, MacAddress.of(dataLayerSource));
        short dataLayerType = packetDataBB.getShort();
        builder.setExact(MatchField.ETH_TYPE, EthType.of(dataLayerType));
            builder.setExact(MatchField.VLAN_VID, OFVlanVidMatch.UNTAGGED);
            builder.setExact(MatchField.VLAN_PCP, VlanPcp.NONE);
        } else {
            scratch = packetDataBB.getShort();
            builder.setExact(MatchField.VLAN_VID, OFVlanVidMatch.ofVlan(0xfff & scratch));
            builder.setExact(MatchField.VLAN_PCP, VlanPcp.of((byte)((0xe000 & scratch) >> 13)));
            dataLayerType = packetDataBB.getShort();
        }
        short networkProtocol;
        int networkSource;
        int networkDestination;
        switch (dataLayerType) {
        case 0x0800:
            scratch = packetDataBB.get();
            scratch = (short) (0xf & scratch);
            scratch = packetDataBB.get();
            builder.setExact(MatchField.IP_ECN, IpEcn.of((byte)(scratch & 0x03)));
            builder.setExact(MatchField.IP_DSCP, IpDscp.of((byte) ((0xfc & scratch) >> 2)));
            packetDataBB.position(packetDataBB.position() + 7);
            networkProtocol = packetDataBB.get();
            builder.setExact(MatchField.IP_PROTO, IpProtocol.of(networkProtocol));
            packetDataBB.position(packetDataBB.position() + 2);
            networkSource = packetDataBB.getInt();
            builder.setExact(MatchField.IPV4_SRC, IPv4Address.of(networkSource));
            networkDestination = packetDataBB.getInt();
            builder.setExact(MatchField.IPV4_DST, IPv4Address.of(networkDestination));
            packetDataBB.position(transportOffset);
            int port;
            switch (networkProtocol) {
            case 0x01:
                short type = U8.f(packetDataBB.get());
                builder.setExact(MatchField.ICMPV4_TYPE, ICMPv4Type.of(type));
                short code = U8.f(packetDataBB.get());
                builder.setExact(MatchField.ICMPV4_CODE, ICMPv4Code.of(code));
                break;
            case 0x06:
                port = packetDataBB.getShort();
                builder.setExact(MatchField.TCP_SRC, TransportPort.of(port));
                port = packetDataBB.getShort();
                builder.setExact(MatchField.TCP_DST, TransportPort.of(port));
                break;
            case 0x11:
                port = packetDataBB.getShort();
                builder.setExact(MatchField.UDP_SRC, TransportPort.of(port));
                port = packetDataBB.getShort();
                builder.setExact(MatchField.UDP_DST, TransportPort.of(port));
                break;
            default:
                break;
            }
            break;
        case 0x0806:
            int arpPos = packetDataBB.position();
            scratch = packetDataBB.getShort(arpPos + 6);
            builder.setExact(MatchField.ARP_OP, ArpOpcode.of(0xff & scratch));
            scratch = packetDataBB.getShort(arpPos + 2);
            if (scratch == 0x800 && packetDataBB.get(arpPos + 5) == 4) {
                networkSource = packetDataBB.getInt(arpPos + 14);
                networkDestination = packetDataBB.getInt(arpPos + 24);
            } else {
                networkSource = 0;
                networkDestination = 0;
            }
            builder.setExact(MatchField.ARP_SPA, IPv4Address.of(networkSource));
            builder.setExact(MatchField.ARP_TPA, IPv4Address.of(networkDestination));
            break;
        default:
            break;
        }
        return builder.build();
    }
}
