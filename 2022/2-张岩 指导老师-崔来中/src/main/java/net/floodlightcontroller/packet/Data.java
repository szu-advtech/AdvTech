package net.floodlightcontroller.packet;
import java.util.Arrays;
public class Data extends BasePacket {
    protected byte[] data;
    public Data() {
    }
    public Data(byte[] data) {
        this.data = data;
    }
    public byte[] getData() {
        return data;
    }
    public Data setData(byte[] data) {
        this.data = data;
        return this;
    }
    public byte[] serialize() {
        return this.data;
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length) {
        this.data = Arrays.copyOfRange(data, offset, offset + length);
        return this;
    }
    @Override
    public int hashCode() {
        final int prime = 1571;
        int result = super.hashCode();
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!super.equals(obj))
            return false;
        if (!(obj instanceof Data))
            return false;
        Data other = (Data) obj;
        if (!Arrays.equals(data, other.data))
            return false;
        return true;
    }
}
