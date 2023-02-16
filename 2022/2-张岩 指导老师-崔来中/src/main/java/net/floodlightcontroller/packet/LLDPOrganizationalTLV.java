package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Arrays;
public class LLDPOrganizationalTLV extends LLDPTLV {
    public static final int OUI_LENGTH = 3;
    public static final int SUBTYPE_LENGTH = 1;
    public static final byte ORGANIZATIONAL_TLV_TYPE = 127;
    public static final int MAX_INFOSTRING_LENGTH = 507;
    protected byte[] oui;
    protected byte subType;
    private byte[] infoString;
    public LLDPOrganizationalTLV() {
        type = ORGANIZATIONAL_TLV_TYPE;
    }
    public LLDPOrganizationalTLV setOUI(byte[] oui) {
        if (oui.length != OUI_LENGTH) {
            throw new IllegalArgumentException("The length of OUI must be " + OUI_LENGTH +
                ", but it is " + oui.length);
        }
        this.oui = Arrays.copyOf(oui, oui.length);
        return this;
    }
    public byte[] getOUI() {
        return Arrays.copyOf(oui, oui.length);
    }
    public LLDPOrganizationalTLV setSubType(byte subType) {
        this.subType = subType;
        return this;
    }
    public byte getSubType() {
        return subType;
    }
    public LLDPOrganizationalTLV setInfoString(byte[] infoString) {
        if (infoString.length > MAX_INFOSTRING_LENGTH) {
            throw new IllegalArgumentException("The length of infoString cannot exceed " + MAX_INFOSTRING_LENGTH);
        }
        this.infoString = Arrays.copyOf(infoString, infoString.length);
        return this;
    }
    public LLDPOrganizationalTLV setInfoString(String infoString) {
        byte[] infoStringBytes = infoString.getBytes(Charset.forName("UTF-8"));
        return setInfoString(infoStringBytes);
    }
    public byte[] getInfoString() {
        return Arrays.copyOf(infoString, infoString.length);
    }
    @Override
    public byte[] serialize() {
        int valueLength = OUI_LENGTH + SUBTYPE_LENGTH + infoString.length;
        value = new byte[valueLength];
        ByteBuffer bb = ByteBuffer.wrap(value);
        bb.put(oui);
        bb.put(subType);
        bb.put(infoString);
        return super.serialize();
    }
    @Override
    public LLDPTLV deserialize(ByteBuffer bb) {
        super.deserialize(bb);
        ByteBuffer optionalField = ByteBuffer.wrap(value);
        byte[] oui = new byte[OUI_LENGTH];
        optionalField.get(oui);
        setOUI(oui);
        setSubType(optionalField.get());
        byte[] infoString = new byte[getLength() - OUI_LENGTH - SUBTYPE_LENGTH];
        optionalField.get(infoString);
        setInfoString(infoString);
        return this;
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!super.equals(obj)) return false;
        if (getClass() != obj.getClass()) return false;
        LLDPOrganizationalTLV other = (LLDPOrganizationalTLV) obj;
        if (!Arrays.equals(infoString, other.infoString)) return false;
        if (!Arrays.equals(oui, other.oui)) return false;
        if (subType != other.subType) return false;
        return true;
    }
}
