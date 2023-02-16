package net.floodlightcontroller.packet;
import java.util.Arrays;
public class DHCPOption {
    protected byte code;
    protected byte length;
    protected byte[] data;
    public byte getCode() {
        return code;
    }
    public DHCPOption setCode(byte code) {
        this.code = code;
        return this;
    }
    public byte getLength() {
        return length;
    }
    public DHCPOption setLength(byte length) {
        this.length = length;
        return this;
    }
    public byte[] getData() {
        return data;
    }
    public DHCPOption setData(byte[] data) {
        this.data = data;
        return this;
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (!(obj instanceof DHCPOption))
            return false;
        DHCPOption other = (DHCPOption) obj;
        if (code != other.code)
            return false;
        if (!Arrays.equals(data, other.data))
            return false;
        if (length != other.length)
            return false;
        return true;
    }
    @Override
    public String toString() {
        return "DHCPOption [code=" + code + ", length=" + length + ", data="
                + Arrays.toString(data) + "]";
    }
}
