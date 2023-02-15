package net.floodlightcontroller.packet;
public interface IPacket {
    public IPacket getPayload();
    public IPacket setPayload(IPacket packet);
    public IPacket getParent();
    public IPacket setParent(IPacket packet);
    public void resetChecksum();
    public byte[] serialize();
    public IPacket deserialize(byte[] data, int offset, int length)
            throws PacketParsingException;
    public Object clone();
}
