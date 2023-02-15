package net.floodlightcontroller.packet;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.MacAddress;
public class DHCP extends BasePacket {
    public static int MIN_HEADER_LENGTH = 240;
    public static byte OPCODE_REQUEST = 0x1;
    public static byte OPCODE_REPLY = 0x2;
    public static byte HWTYPE_ETHERNET = 0x1;
    public enum DHCPOptionCode {
        OptionCode_SubnetMask           ((byte)1),
        OptionCode_Hostname             ((byte)12),
        OptionCode_RequestedIP          ((byte)50),
        OptionCode_LeaseTime            ((byte)51),
        OptionCode_MessageType          ((byte)53),
        OptionCode_DHCPServerIp         ((byte)54),
        OptionCode_RequestedParameters  ((byte)55),
        OptionCode_RenewalTime          ((byte)58),
        OPtionCode_RebindingTime        ((byte)59),
        OptionCode_ClientID             ((byte)61),
        OptionCode_END                  ((byte)255);
        protected byte value;
        private DHCPOptionCode(byte value) {
            this.value = value;
        }
        public byte getValue() {
            return value;
        }
    }
    protected byte opCode;
    protected byte hardwareType;
    protected byte hardwareAddressLength;
    protected byte hops;
    protected int transactionId;
    protected short seconds;
    protected short flags;
    protected IPv4Address clientIPAddress;
    protected IPv4Address yourIPAddress;
    protected IPv4Address serverIPAddress;
    protected IPv4Address gatewayIPAddress;
    protected IPv4Address requestedIP;
    protected MacAddress clientHardwareAddress;
    protected String serverName;
    protected String bootFileName;
    protected List<DHCPOption> options = new ArrayList<DHCPOption>();
    public byte getOpCode() {
        return opCode;
    }
    public IPv4Address getRequestIP() {
		return requestedIP;
	}
    public DHCP setOpCode(byte opCode) {
        this.opCode = opCode;
        return this;
    }
    public byte getHardwareType() {
        return hardwareType;
    }
    public DHCP setHardwareType(byte hardwareType) {
        this.hardwareType = hardwareType;
        return this;
    }
    public byte getHardwareAddressLength() {
        return hardwareAddressLength;
    }
    public DHCP setHardwareAddressLength(byte hardwareAddressLength) {
        this.hardwareAddressLength = hardwareAddressLength;
        return this;
    }
    public byte getHops() {
        return hops;
    }
    public DHCP setHops(byte hops) {
        this.hops = hops;
        return this;
    }
    public int getTransactionId() {
        return transactionId;
    }
    public DHCP setTransactionId(int transactionId) {
        this.transactionId = transactionId;
        return this;
    }
    public short getSeconds() {
        return seconds;
    }
    public DHCP setSeconds(short seconds) {
        this.seconds = seconds;
        return this;
    }
    public short getFlags() {
        return flags;
    }
    public DHCP setFlags(short flags) {
        this.flags = flags;
        return this;
    }
    public IPv4Address getClientIPAddress() {
        return clientIPAddress;
    }
    public DHCP setClientIPAddress(IPv4Address clientIPAddress) {
        this.clientIPAddress = clientIPAddress;
        return this;
    }
    public IPv4Address getYourIPAddress() {
        return yourIPAddress;
    }
    public DHCP setYourIPAddress(IPv4Address yourIPAddress) {
        this.yourIPAddress = yourIPAddress;
        return this;
    }
    public IPv4Address getServerIPAddress() {
        return serverIPAddress;
    }
    public DHCP setServerIPAddress(IPv4Address serverIPAddress) {
        this.serverIPAddress = serverIPAddress;
        return this;
    }
    public IPv4Address getGatewayIPAddress() {
        return gatewayIPAddress;
    }
    public DHCP setGatewayIPAddress(IPv4Address gatewayIPAddress) {
        this.gatewayIPAddress = gatewayIPAddress;
        return this;
    }
    public MacAddress getClientHardwareAddress() {
        return clientHardwareAddress;
    }
    public DHCP setClientHardwareAddress(MacAddress clientHardwareAddress) {
        this.clientHardwareAddress = clientHardwareAddress;
        return this;
    }
    public DHCPOption getOption(DHCPOptionCode optionCode) {
        for (DHCPOption opt : options) {
            if (opt.code == optionCode.value)
                return opt;
        }
        return null;
    }
    public List<DHCPOption> getOptions() {
        return options;
    }
    public DHCP setOptions(List<DHCPOption> options) {
        this.options = options;
        return this;
    }
    public DHCPPacketType getPacketType() {
        ListIterator<DHCPOption> lit = options.listIterator();
        while (lit.hasNext()) {
            DHCPOption option = lit.next();
            if (option.getCode() == 53) {
                return DHCPPacketType.getType(option.getData()[0]);
            }
        }
        return null;
    }
    public String getServerName() {
        return serverName;
    }
    public DHCP setServerName(String serverName) {
        this.serverName = serverName;
        return this;
    }
    public String getBootFileName() {
        return bootFileName;
    }
    public DHCP setBootFileName(String bootFileName) {
        this.bootFileName = bootFileName;
        return this;
    }
    @Override
    public byte[] serialize() {
        resetChecksum();
        int optionsLength = 0;
        for (DHCPOption option : this.options) {
            if (option.getCode() == 0 || option.getCode() == ((byte)255)) {
                optionsLength += 1;
            } else {
                optionsLength += 2 + (0xff & option.getLength());
            }
        }
        int optionsPadLength = 0;
        if (optionsLength < 60)
            optionsPadLength = 60 - optionsLength;
        byte[] data = new byte[240+optionsLength+optionsPadLength];
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.put(this.opCode);
        bb.put(this.hardwareType);
        bb.put(this.hardwareAddressLength);
        bb.put(this.hops);
        bb.putInt(this.transactionId);
        bb.putShort(this.seconds);
        bb.putShort(this.flags);
        bb.putInt(this.clientIPAddress.getInt());
        bb.putInt(this.yourIPAddress.getInt());
        bb.putInt(this.serverIPAddress.getInt());
        bb.putInt(this.gatewayIPAddress.getInt());
        bb.put(this.clientHardwareAddress.getBytes());
        if (this.clientHardwareAddress.getLength() < 16) {
            for (int i = 0; i < (16 - this.clientHardwareAddress.getLength()); ++i) {
                bb.put((byte) 0x0);
            }
        }
        writeString(this.serverName, bb, 64);
        writeString(this.bootFileName, bb, 128);
        bb.put((byte) 0x63);
        bb.put((byte) 0x82);
        bb.put((byte) 0x53);
        bb.put((byte) 0x63);
        for (DHCPOption option : this.options) {
            int code = option.getCode() & 0xff;
            bb.put((byte) code);
            if ((code != 0) && (code != 255)) {
                bb.put(option.getLength());
                bb.put(option.getData());
            }
        }
        return data;
    }
    protected void writeString(String string, ByteBuffer bb, int maxLength) {
        if (string == null) {
            for (int i = 0; i < maxLength; ++i) {
                bb.put((byte) 0x0);
            }
        } else {
            byte[] bytes = null;
            try {
                 bytes = string.getBytes("ascii");
            } catch (UnsupportedEncodingException e) {
                throw new RuntimeException("Failure encoding server name", e);
            }
            int writeLength = bytes.length;
            if (writeLength > maxLength) {
                writeLength = maxLength;
            }
            bb.put(bytes, 0, writeLength);
            for (int i = writeLength; i < maxLength; ++i) {
                bb.put((byte) 0x0);
            }
        }
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length) {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
        if (bb.remaining() < MIN_HEADER_LENGTH) {
            return this;
        }
        this.opCode = bb.get();
        this.hardwareType = bb.get();
        this.hardwareAddressLength = bb.get();
        this.hops = bb.get();
        this.transactionId = bb.getInt();
        this.seconds = bb.getShort();
        this.flags = bb.getShort();
        this.clientIPAddress = IPv4Address.of(bb.getInt());
        this.yourIPAddress = IPv4Address.of(bb.getInt());
        this.serverIPAddress = IPv4Address.of(bb.getInt());
        this.gatewayIPAddress = IPv4Address.of(bb.getInt());
        this.requestedIP = null;
        int hardwareAddressLength = 0xff & this.hardwareAddressLength;
        byte[] tmpMac = new byte[hardwareAddressLength];
        bb.get(tmpMac);
        for (int i = hardwareAddressLength; i < 16; ++i)
            bb.get();
        this.serverName = readString(bb, 64);
        this.bootFileName = readString(bb, 128);
        bb.get();
        bb.get();
        bb.get();
        bb.get();
        while (bb.hasRemaining()) {
            DHCPOption option = new DHCPOption();
            option.setCode((byte) code);
            if (code == 0) {
                continue;
            } else if (code != 255) {
                if (bb.hasRemaining()) {
                    option.setLength((byte) l);
                    if (bb.remaining() >= l) {
                        byte[] optionData = new byte[l];
                        bb.get(optionData);
                        option.setData(optionData);
                    } else {
                        code = 0xff;
                        option.setCode((byte)code);
                        option.setLength((byte) 0);
                    }
                } else {
                    code = 0xff;
                    option.setCode((byte)code);
                    option.setLength((byte) 0);
                }
            }
            this.options.add(option);
            if (code == 255) {
                break;
            }
            if(code == 50){
            	this.requestedIP = IPv4Address.of(option.getData());
            }
        }
        return this;
    }
    protected String readString(ByteBuffer bb, int maxLength) {
        byte[] bytes = new byte[maxLength];
        bb.get(bytes);
        String result = null;
        try {
            result = new String(bytes, "ascii").trim();
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException("Failure decoding string", e);
        }
        return result;
    }
}
