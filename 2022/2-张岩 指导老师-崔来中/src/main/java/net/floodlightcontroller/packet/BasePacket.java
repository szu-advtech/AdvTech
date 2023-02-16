package net.floodlightcontroller.packet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public abstract class BasePacket implements IPacket {
    public static final Logger log = LoggerFactory.getLogger(BasePacket.class);
    protected IPacket parent;
    protected IPacket payload;
    @Override
    public IPacket getParent() {
        return parent;
    }
    @Override
    public IPacket setParent(IPacket parent) {
        this.parent = parent;
        return this;
    }
    @Override
    public IPacket getPayload() {
        return payload;
    }
    @Override
    public IPacket setPayload(IPacket payload) {
        this.payload = payload;
        return this;
    }
    @Override
    public void resetChecksum() {
        if (this.parent != null)
            this.parent.resetChecksum();
    }
    @Override
    public int hashCode() {
        final int prime = 6733;
        int result = 1;
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (!(obj instanceof BasePacket))
            return false;
        BasePacket other = (BasePacket) obj;
        if (payload == null) {
            if (other.payload != null)
                return false;
        } else if (!payload.equals(other.payload))
            return false;
        return true;
    }
    @Override
    public Object clone() {
        IPacket pkt;
        try {
            pkt = this.getClass().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Could not clone packet");
        }
        byte[] data = this.serialize();
        try {
            pkt.deserialize(this.serialize(), 0, data.length);
        } catch (PacketParsingException e) {
            return new Data(data);
        }
        pkt.setParent(this.parent);
        return pkt;
    }
}
