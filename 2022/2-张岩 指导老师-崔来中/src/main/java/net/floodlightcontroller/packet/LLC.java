package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
public class LLC extends BasePacket {
    private byte dsap = 0;
    private byte ssap = 0;
    private byte ctrl = 0;
    public byte getDsap() {
        return dsap;
    }
    public void setDsap(byte dsap) {
        this.dsap = dsap;
    }
    public byte getSsap() {
        return ssap;
    }
    public void setSsap(byte ssap) {
        this.ssap = ssap;
    }
    public byte getCtrl() {
        return ctrl;
    }
    public void setCtrl(byte ctrl) {
        this.ctrl = ctrl;
    }
    @Override
    public byte[] serialize() {
        byte[] data = new byte[3];
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.put(dsap);
        bb.put(ssap);
        bb.put(ctrl);
        return data;
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length) {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
        dsap = bb.get();
        ssap = bb.get();
        ctrl = bb.get();
        return this;
    }
}
