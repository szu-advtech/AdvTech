package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.Arrays;
public class LLDPTLV {
    protected byte type;
    protected short length;
    protected byte[] value;
    public byte getType() {
        return type;
    }
    public LLDPTLV setType(byte type) {
        this.type = type;
        return this;
    }
    public short getLength() {
        return length;
    }
    public LLDPTLV setLength(short length) {
        this.length = length;
        return this;
    }
    public byte[] getValue() {
        return value;
    }
    public LLDPTLV setValue(byte[] value) {
        this.value = value;
        return this;
    }
    public byte[] serialize() {
        short scratch = (short) (((0x7f & this.type) << 9) | (0x1ff & this.length));
        byte[] data = new byte[2+this.length];
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.putShort(scratch);
        if (this.value != null)
            bb.put(this.value);
        return data;
    }
    public LLDPTLV deserialize(ByteBuffer bb) {
        short sscratch;
        sscratch = bb.getShort();
        this.type = (byte) ((sscratch >> 9) & 0x7f);
        this.length = (short) (sscratch & 0x1ff);
        if (this.length > 0) {
            this.value = new byte[this.length];
            if (bb.remaining() < this.length)
                return null;
            bb.get(this.value);
        }
        return this;
    }
    @Override
    public int hashCode() {
        final int prime = 1423;
        int result = 1;
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (!(obj instanceof LLDPTLV))
            return false;
        LLDPTLV other = (LLDPTLV) obj;
        if (length != other.length)
            return false;
        if (type != other.type)
            return false;
        if (!Arrays.equals(value, other.value))
            return false;
        return true;
    }
    @Override
    public String toString() {
        String str = "type=" + Integer.toString(this.type, 16).toUpperCase()
               + " length=" + this.length
               + " value=";
        for (byte b : this.value)
            str+= Integer.toString(b, 16).toUpperCase();
        return str;
    }
}
