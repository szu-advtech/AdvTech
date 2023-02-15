package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.projectfloodlight.openflow.types.EthType;
public class LLDP extends BasePacket {
    protected LLDPTLV chassisId;
    protected LLDPTLV portId;
    protected LLDPTLV ttl;
    protected List<LLDPTLV> optionalTLVList;
    protected EthType ethType;
    public LLDP() {
        this.optionalTLVList = new ArrayList<LLDPTLV>();
        this.ethType = EthType.LLDP;
    }
    public LLDPTLV getChassisId() {
        return chassisId;
    }
    public LLDP setChassisId(LLDPTLV chassisId) {
        this.chassisId = chassisId;
        return this;
    }
    public LLDPTLV getPortId() {
        return portId;
    }
    public LLDP setPortId(LLDPTLV portId) {
        this.portId = portId;
        return this;
    }
    public LLDPTLV getTtl() {
        return ttl;
    }
    public LLDP setTtl(LLDPTLV ttl) {
        this.ttl = ttl;
        return this;
    }
    public List<LLDPTLV> getOptionalTLVList() {
        return optionalTLVList;
    }
    public LLDP setOptionalTLVList(List<LLDPTLV> optionalTLVList) {
        this.optionalTLVList = optionalTLVList;
        return this;
    }
    @Override
    public byte[] serialize() {
        int length = 2+this.chassisId.getLength() + 2+this.portId.getLength() +
            2+this.ttl.getLength() + 2;
        for (LLDPTLV tlv : this.optionalTLVList) {
            if (tlv != null)
                length += 2 + tlv.getLength();
        }
        byte[] data = new byte[length];
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.put(this.chassisId.serialize());
        bb.put(this.portId.serialize());
        bb.put(this.ttl.serialize());
        for (LLDPTLV tlv : this.optionalTLVList) {
            if (tlv != null) bb.put(tlv.serialize());
        }
        if (this.parent != null && this.parent instanceof Ethernet)
            ((Ethernet)this.parent).setEtherType(ethType);
        return data;
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length) {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
        LLDPTLV tlv;
        do {
            tlv = new LLDPTLV().deserialize(bb);
            if (tlv == null)
                break;
            switch (tlv.getType()) {
                case 0x0:
                    break;
                case 0x1:
                    this.chassisId = tlv;
                    break;
                case 0x2:
                    this.portId = tlv;
                    break;
                case 0x3:
                    this.ttl = tlv;
                    break;
                default:
                    this.optionalTLVList.add(tlv);
                    break;
            }
        } while (tlv.getType() != 0 && bb.hasRemaining());
        return this;
    }
    @Override
    public int hashCode() {
        final int prime = 883;
        int result = super.hashCode();
                + ((chassisId == null) ? 0 : chassisId.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!super.equals(obj))
            return false;
        if (!(obj instanceof LLDP))
            return false;
        LLDP other = (LLDP) obj;
        if (chassisId == null) {
            if (other.chassisId != null)
                return false;
        } else if (!chassisId.equals(other.chassisId))
            return false;
        if (!optionalTLVList.equals(other.optionalTLVList))
            return false;
        if (portId == null) {
            if (other.portId != null)
                return false;
        } else if (!portId.equals(other.portId))
            return false;
        if (ttl == null) {
            if (other.ttl != null)
                return false;
        } else if (!ttl.equals(other.ttl))
            return false;
        return true;
    }
    @Override
    public String toString() {
        String str = "";
        str += "chassisId=" + ((this.chassisId == null) ? "null" : this.chassisId.toString());
        str += " portId=" + ((this.portId == null) ? "null" : this.portId.toString());
        str += " ttl=" + ((this.ttl == null) ? "null" : this.ttl.toString());
        str += " etherType=" + ethType.toString();
        str += " optionalTlvList=[";
        if (this.optionalTLVList != null) {
            for (LLDPTLV l : optionalTLVList) str += l.toString() + ", ";
        }
        str += "]";
        return str;
    }
}
