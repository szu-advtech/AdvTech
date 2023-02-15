package org.sdnplatform.sync.internal.util;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import org.projectfloodlight.openflow.util.HexString;
public final class ByteArray implements Serializable {
    private static final long serialVersionUID = 1L;
    public static final ByteArray EMPTY = new ByteArray();
    private final byte[] underlying;
    public ByteArray(byte... underlying) {
        this.underlying = underlying;
    }
    public byte[] get() {
        return underlying;
    }
    @Override
    public int hashCode() {
        return Arrays.hashCode(underlying);
    }
    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(!(obj instanceof ByteArray))
            return false;
        ByteArray other = (ByteArray) obj;
        return Arrays.equals(underlying, other.underlying);
    }
    @Override
    public String toString() {
        return Arrays.toString(underlying);
    }
    public static Iterable<String> toHexStrings(Iterable<ByteArray> arrays) {
        ArrayList<String> ret = new ArrayList<String>();
        for(ByteArray array: arrays)
            ret.add(HexString.toHexString(array.get()));
        return ret;
    }
    public int length() {
        return underlying.length;
    }
}
