package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
public class BPDU extends BasePacket {
    public enum BPDUType {
        CONFIG,
        TOPOLOGY_CHANGE;
    }
    private LLC llcHeader;
    private short protocolId = 0;
    private byte version = 0;
    private byte type;
    private byte flags;
    private byte[] rootBridgeId;
    private int rootPathCost;
    public BPDU(BPDUType type) {
        rootBridgeId = new byte[8];
        senderBridgeId = new byte[8];
        llcHeader = new LLC();
        llcHeader.setDsap((byte) 0x42);
        llcHeader.setSsap((byte) 0x42);
        llcHeader.setCtrl((byte) 0x03);
        switch(type) {
            case CONFIG:
                this.type = 0x0;
                break;
            case TOPOLOGY_CHANGE:
                break;
            default:
                this.type = 0;
                break;
        }
    }
    @Override
    public byte[] serialize() {
        byte[] data;
        if (type == 0x0) { 
            data = new byte[38];
        } else {
        }
        ByteBuffer bb = ByteBuffer.wrap(data);
        byte[] llc = llcHeader.serialize();
        bb.put(llc, 0, llc.length);
        bb.putShort(protocolId);
        bb.put(version);
        bb.put(type);
        if (type == 0x0) {
            bb.put(flags);
            bb.put(rootBridgeId, 0, rootBridgeId.length);
            bb.putInt(rootPathCost);
            bb.put(senderBridgeId, 0, senderBridgeId.length);
            bb.putShort(portId);
            bb.putShort(messageAge);
            bb.putShort(maxAge);
            bb.putShort(helloTime);
            bb.putShort(forwardDelay);
        }
        return data;
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length) {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
        llcHeader.deserialize(data, offset, 3);
        this.protocolId = bb.getShort();
        this.version = bb.get();
        this.type = bb.get();
        if (this.type == 0x0) {
            this.flags = bb.get();
            bb.get(rootBridgeId, 0, 6);
            this.rootPathCost = bb.getInt();
            bb.get(this.senderBridgeId, 0, 6);
            this.portId = bb.getShort();
            this.messageAge = bb.getShort();
            this.maxAge = bb.getShort();
            this.helloTime = bb.getShort();
            this.forwardDelay = bb.getShort();
        }
        return this;
    }
    public long getDestMac() {
        return destMac;
    }
}
