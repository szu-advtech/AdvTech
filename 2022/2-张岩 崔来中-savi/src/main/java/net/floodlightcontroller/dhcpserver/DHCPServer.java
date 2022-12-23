package net.floodlightcontroller.dhcpserver;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPacketOut;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFPort;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.forwarding.Forwarding;
import net.floodlightcontroller.packet.DHCP.DHCPOptionCode;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.UDP;
import net.floodlightcontroller.packet.DHCP;
import net.floodlightcontroller.packet.DHCPOption;
public class DHCPServer implements IOFMessageListener, IFloodlightModule  {
	protected static Logger log;
	protected static IFloodlightProviderService floodlightProvider;
	protected static IOFSwitchService switchService;
	private static ScheduledThreadPoolExecutor leasePoliceDispatcher;
	private static Runnable leasePolicePatrol;
	private static volatile DHCPPool theDHCPPool;
	private static MacAddress CONTROLLER_MAC;
	private static IPv4Address CONTROLLER_IP;
	private static IPv4Address DHCP_SERVER_SUBNET_MASK;
	private static IPv4Address DHCP_SERVER_BROADCAST_IP;
	private static IPv4Address DHCP_SERVER_IP_START;
	private static IPv4Address DHCP_SERVER_IP_STOP;
	private static IPv4Address DHCP_SERVER_ROUTER_IP = null;
	private static byte[] DHCP_SERVER_NTP_IP_LIST = null;
	private static byte[] DHCP_SERVER_DNS_IP_LIST = null;
	private static byte[] DHCP_SERVER_DN = null;
	private static byte[] DHCP_SERVER_IP_FORWARDING = null;
	private static int DHCP_SERVER_DEFAULT_LEASE_TIME_SECONDS;
	private static int DHCP_SERVER_HOLD_LEASE_TIME_SECONDS;
	private static long DHCP_SERVER_LEASE_POLICE_PATROL_PERIOD_SECONDS;
	public static byte DHCP_OPCODE_REQUEST = intToBytes(1)[0];
	public static byte DHCP_OPCODE_REPLY = intToBytes(2)[0];
	public static byte[] DHCP_MSG_TYPE_DISCOVER = intToBytesSizeOne(1);
	public static byte[] DHCP_MSG_TYPE_OFFER = intToBytesSizeOne(2);
	public static byte[] DHCP_MSG_TYPE_REQUEST = intToBytesSizeOne(3);
	public static byte[] DHCP_MSG_TYPE_DECLINE = intToBytesSizeOne(4);
	public static byte[] DHCP_MSG_TYPE_ACK = intToBytesSizeOne(5);
	public static byte[] DHCP_MSG_TYPE_NACK = intToBytesSizeOne(6);
	public static byte[] DHCP_MSG_TYPE_RELEASE = intToBytesSizeOne(7);
	public static byte[] DHCP_MSG_TYPE_INFORM = intToBytesSizeOne(8);
	public static byte DHCP_REQ_PARAM_OPTION_CODE_SN = intToBytes(1)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_ROUTER = intToBytes(3)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_DNS = intToBytes(6)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_DN = intToBytes(15)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_IP_FORWARDING = intToBytes(19)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_BROADCAST_IP = intToBytes(28)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_NTP_IP = intToBytes(42)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_NET_BIOS_NAME_IP = intToBytes(44)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_NET_BIOS_DDS_IP = intToBytes(45)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_NET_BIOS_NODE_TYPE = intToBytes(46)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_NET_BIOS_SCOPE_ID = intToBytes(47)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_REQUESTED_IP = intToBytes(50)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME = intToBytes(51)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_MSG_TYPE = intToBytes(53)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER = intToBytes(54)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_REQUESTED_PARAMTERS = intToBytes(55)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME = intToBytes(58)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME = intToBytes(59)[0];
	public static byte DHCP_REQ_PARAM_OPTION_CODE_END = intToBytes(255)[0];
	public static final MacAddress BROADCAST_MAC = MacAddress.BROADCAST;
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l = 
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		floodlightProvider = context.getServiceImpl(IFloodlightProviderService.class);
		switchService = context.getServiceImpl(IOFSwitchService.class);
		log = LoggerFactory.getLogger(DHCPServer.class);
	}
	@Override
	public void startUp(FloodlightModuleContext context) {
		floodlightProvider.addOFMessageListener(OFType.PACKET_IN, this);
		Map<String, String> configOptions = context.getConfigParams(this);
		try {
			DHCP_SERVER_SUBNET_MASK = IPv4Address.of(configOptions.get("subnet-mask"));
			DHCP_SERVER_IP_START = IPv4Address.of(configOptions.get("lower-ip-range"));
			DHCP_SERVER_IP_STOP = IPv4Address.of(configOptions.get("upper-ip-range"));
			DHCP_SERVER_ADDRESS_SPACE_SIZE = DHCP_SERVER_IP_STOP.getInt() - DHCP_SERVER_IP_START.getInt() + 1;
			DHCP_SERVER_BROADCAST_IP = IPv4Address.of(configOptions.get("broadcast-address"));
			DHCP_SERVER_ROUTER_IP = IPv4Address.of(configOptions.get("router"));
			DHCP_SERVER_DN = configOptions.get("domain-name").getBytes();
			DHCP_SERVER_DEFAULT_LEASE_TIME_SECONDS = Integer.parseInt(configOptions.get("default-lease-time"));
			DHCP_SERVER_HOLD_LEASE_TIME_SECONDS = Integer.parseInt(configOptions.get("hold-lease-time"));
			DHCP_SERVER_RENEWAL_TIME_SECONDS = (int) (DHCP_SERVER_DEFAULT_LEASE_TIME_SECONDS / 2.0);
			DHCP_SERVER_LEASE_POLICE_PATROL_PERIOD_SECONDS = Long.parseLong(configOptions.get("lease-gc-period"));
			DHCP_SERVER_IP_FORWARDING = intToBytesSizeOne(Integer.parseInt(configOptions.get("ip-forwarding")));
			CONTROLLER_MAC = MacAddress.of(configOptions.get("controller-mac"));
			CONTROLLER_IP = IPv4Address.of(configOptions.get("controller-ip"));
			DHCP_SERVER_DHCP_SERVER_IP = CONTROLLER_IP;
		} catch(IllegalArgumentException ex) {
			log.error("Incorrect DHCP Server configuration options", ex);
			throw ex;
		} catch(NullPointerException ex) {
			log.error("Incorrect DHCP Server configuration options", ex);
			throw ex;
		}
		theDHCPPool = new DHCPPool(DHCP_SERVER_IP_START, DHCP_SERVER_ADDRESS_SPACE_SIZE, log);
		String staticAddresses = configOptions.get("reserved-static-addresses");
		if (staticAddresses != null) {
			int i;
			String[] macIpSplit;
			int ipPos, macPos;
			for (i = 0; i < macIpCouples.length; i++) {
				if (macIpSplit[0].length() > macIpSplit[1].length()) {
					macPos = 0;
					ipPos = 1;
				} else {
					macPos = 1;
					ipPos = 0;
				}
				if (theDHCPPool.configureFixedIPLease(IPv4Address.of(macIpSplit[ipPos]), MacAddress.of(macIpSplit[macPos]))) {
					String ip = theDHCPPool.getDHCPbindingFromIPv4(IPv4Address.of(macIpSplit[ipPos])).getIPv4Address().toString();
					String mac = theDHCPPool.getDHCPbindingFromIPv4(IPv4Address.of(macIpSplit[ipPos])).getMACAddress().toString();
					log.info("Configured fixed address of " + ip + " for device " + mac);
				} else {
					log.error("Could not configure fixed address " + macIpSplit[ipPos] + " for device " + macIpSplit[macPos]);
				}
			}
		}
		String dnses = configOptions.get("domain-name-servers");
		String ntps = configOptions.get("ntp-servers");
		if (dnses != null) {
		}
		if (ntps != null) {
		}
		leasePoliceDispatcher = new ScheduledThreadPoolExecutor(1);
		leasePolicePatrol = new DHCPLeasePolice();
		leasePoliceDispatcher.scheduleAtFixedRate(leasePolicePatrol, 10, 
				DHCP_SERVER_LEASE_POLICE_PATROL_PERIOD_SECONDS, TimeUnit.SECONDS);
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		return null;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		return null;
	}
	@Override
	public String getName() {
		return DHCPServer.class.getSimpleName();
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
		return false;
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
		if (type == OFType.PACKET_IN && name.equals(Forwarding.class.getSimpleName())) {
			return true;
		} else {
			return false;
		}
	}
	public static byte[] intToBytes(int integer) {
		byte[] bytes = new byte[4];
		bytes[3] = (byte) (integer >> 24);
		bytes[2] = (byte) (integer >> 16);
		bytes[1] = (byte) (integer >> 8);
		bytes[0] = (byte) (integer);
		return bytes;
	}
	public static byte[] intToBytesSizeOne(int integer) {
		byte[] bytes = new byte[1];
		bytes[0] = (byte) (integer);
		return bytes;
	}
	public void sendDHCPOffer(IOFSwitch sw, OFPort inPort, MacAddress chaddr, IPv4Address dstIPAddr, 
			IPv4Address yiaddr, IPv4Address giaddr, int xid, ArrayList<Byte> requestOrder) {
		OFPacketOut.Builder DHCPOfferPacket = sw.getOFFactory().buildPacketOut();
		DHCPOfferPacket.setBufferId(OFBufferId.NO_BUFFER);
		Ethernet ethDHCPOffer = new Ethernet();
		ethDHCPOffer.setSourceMACAddress(CONTROLLER_MAC);
		ethDHCPOffer.setDestinationMACAddress(chaddr);
		ethDHCPOffer.setEtherType(EthType.IPv4);
		IPv4 ipv4DHCPOffer = new IPv4();
		if (dstIPAddr.equals(IPv4Address.NONE)) {
			ipv4DHCPOffer.setDestinationAddress(BROADCAST_IP);
			ipv4DHCPOffer.setDestinationAddress(dstIPAddr);
		}
		ipv4DHCPOffer.setSourceAddress(CONTROLLER_IP);
		ipv4DHCPOffer.setProtocol(IpProtocol.UDP);
		ipv4DHCPOffer.setTtl((byte) 64);
		UDP udpDHCPOffer = new UDP();
		udpDHCPOffer.setDestinationPort(UDP.DHCP_CLIENT_PORT);
		udpDHCPOffer.setSourcePort(UDP.DHCP_SERVER_PORT);
		DHCP dhcpDHCPOffer = new DHCP();
		dhcpDHCPOffer.setOpCode(DHCP_OPCODE_REPLY);
		dhcpDHCPOffer.setHardwareType((byte) 1);
		dhcpDHCPOffer.setHardwareAddressLength((byte) 6);
		dhcpDHCPOffer.setHops((byte) 0);
		dhcpDHCPOffer.setTransactionId(xid);
		dhcpDHCPOffer.setSeconds((short) 0);
		dhcpDHCPOffer.setFlags((short) 0);
		dhcpDHCPOffer.setClientIPAddress(UNASSIGNED_IP);
		dhcpDHCPOffer.setYourIPAddress(yiaddr);
		dhcpDHCPOffer.setServerIPAddress(CONTROLLER_IP);
		dhcpDHCPOffer.setGatewayIPAddress(giaddr);
		dhcpDHCPOffer.setClientHardwareAddress(chaddr);
		List<DHCPOption> dhcpOfferOptions = new ArrayList<DHCPOption>();
		DHCPOption newOption;
		newOption = new DHCPOption();
		newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_MSG_TYPE);
		newOption.setData(DHCP_MSG_TYPE_OFFER);
		newOption.setLength((byte) 1);
		dhcpOfferOptions.add(newOption);
		for (Byte specificRequest : requestOrder) {
			if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_SN) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_SN);
				newOption.setData(DHCP_SERVER_SUBNET_MASK.getBytes());
				newOption.setLength((byte) 4);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_ROUTER) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_ROUTER);
				newOption.setData(DHCP_SERVER_ROUTER_IP.getBytes());
				newOption.setLength((byte) 4);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_DN) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_DN);
				newOption.setData(DHCP_SERVER_DN);
				newOption.setLength((byte) DHCP_SERVER_DN.length);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_DNS) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_DNS);
				newOption.setData(DHCP_SERVER_DNS_IP_LIST);
				newOption.setLength((byte) DHCP_SERVER_DNS_IP_LIST.length);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_BROADCAST_IP) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_BROADCAST_IP);
				newOption.setData(DHCP_SERVER_BROADCAST_IP.getBytes());
				newOption.setLength((byte) 4);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER);
				newOption.setData(DHCP_SERVER_DHCP_SERVER_IP.getBytes());
				newOption.setLength((byte) 4);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME);
				newOption.setData(intToBytes(DHCP_SERVER_DEFAULT_LEASE_TIME_SECONDS));
				newOption.setLength((byte) 4);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_NTP_IP) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_NTP_IP);
				newOption.setData(DHCP_SERVER_NTP_IP_LIST);
				newOption.setLength((byte) DHCP_SERVER_NTP_IP_LIST.length);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME);
				newOption.setData(intToBytes(DHCP_SERVER_REBIND_TIME_SECONDS));
				newOption.setLength((byte) 4);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME);
				newOption.setData(intToBytes(DHCP_SERVER_RENEWAL_TIME_SECONDS));
				newOption.setLength((byte) 4);
				dhcpOfferOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_IP_FORWARDING) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_IP_FORWARDING);
				newOption.setData(DHCP_SERVER_IP_FORWARDING);
				newOption.setLength((byte) 1);
				dhcpOfferOptions.add(newOption);
			} else {
			}
		}
		newOption = new DHCPOption();
		newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_END);
		newOption.setLength((byte) 0);
		dhcpOfferOptions.add(newOption);
		dhcpDHCPOffer.setOptions(dhcpOfferOptions);
		ethDHCPOffer.setPayload(ipv4DHCPOffer.setPayload(udpDHCPOffer.setPayload(dhcpDHCPOffer)));
		DHCPOfferPacket.setInPort(OFPort.ANY);
		List<OFAction> actions = new ArrayList<OFAction>(1);
		actions.add(sw.getOFFactory().actions().output(inPort, 0xffFFffFF));
		DHCPOfferPacket.setActions(actions);
		DHCPOfferPacket.setData(ethDHCPOffer.serialize());
		log.debug("Sending DHCP OFFER");
		sw.write(DHCPOfferPacket.build());
	}
	public void sendDHCPAck(IOFSwitch sw, OFPort inPort, MacAddress chaddr, IPv4Address dstIPAddr, 
			IPv4Address yiaddr, IPv4Address giaddr, int xid, ArrayList<Byte> requestOrder) {
		OFPacketOut.Builder DHCPACKPacket = sw.getOFFactory().buildPacketOut();
		DHCPACKPacket.setBufferId(OFBufferId.NO_BUFFER);
		Ethernet ethDHCPAck = new Ethernet();
		ethDHCPAck.setSourceMACAddress(CONTROLLER_MAC);
		ethDHCPAck.setDestinationMACAddress(chaddr);
		ethDHCPAck.setEtherType(EthType.IPv4);
		IPv4 ipv4DHCPAck = new IPv4();
		if (dstIPAddr.equals(IPv4Address.NONE)) {
			ipv4DHCPAck.setDestinationAddress(BROADCAST_IP);
			ipv4DHCPAck.setDestinationAddress(dstIPAddr);
		}
		ipv4DHCPAck.setSourceAddress(CONTROLLER_IP);
		ipv4DHCPAck.setProtocol(IpProtocol.UDP);
		ipv4DHCPAck.setTtl((byte) 64);
		UDP udpDHCPAck = new UDP();
		udpDHCPAck.setDestinationPort(UDP.DHCP_CLIENT_PORT);
		udpDHCPAck.setSourcePort(UDP.DHCP_SERVER_PORT);
		DHCP dhcpDHCPAck = new DHCP();
		dhcpDHCPAck.setOpCode(DHCP_OPCODE_REPLY);
		dhcpDHCPAck.setHardwareType((byte) 1);
		dhcpDHCPAck.setHardwareAddressLength((byte) 6);
		dhcpDHCPAck.setHops((byte) 0);
		dhcpDHCPAck.setTransactionId(xid);
		dhcpDHCPAck.setSeconds((short) 0);
		dhcpDHCPAck.setFlags((short) 0);
		dhcpDHCPAck.setClientIPAddress(UNASSIGNED_IP);
		dhcpDHCPAck.setYourIPAddress(yiaddr);
		dhcpDHCPAck.setServerIPAddress(CONTROLLER_IP);
		dhcpDHCPAck.setGatewayIPAddress(giaddr);
		dhcpDHCPAck.setClientHardwareAddress(chaddr);
		List<DHCPOption> dhcpAckOptions = new ArrayList<DHCPOption>();
		DHCPOption newOption;
		newOption = new DHCPOption();
		newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_MSG_TYPE);
		newOption.setData(DHCP_MSG_TYPE_ACK);
		newOption.setLength((byte) 1);
		dhcpAckOptions.add(newOption);
		for (Byte specificRequest : requestOrder) {
			if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_SN) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_SN);
				newOption.setData(DHCP_SERVER_SUBNET_MASK.getBytes());
				newOption.setLength((byte) 4);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_ROUTER) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_ROUTER);
				newOption.setData(DHCP_SERVER_ROUTER_IP.getBytes());
				newOption.setLength((byte) 4);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_DN) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_DN);
				newOption.setData(DHCP_SERVER_DN);
				newOption.setLength((byte) DHCP_SERVER_DN.length);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_DNS) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_DNS);
				newOption.setData(DHCP_SERVER_DNS_IP_LIST);
				newOption.setLength((byte) DHCP_SERVER_DNS_IP_LIST.length);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_BROADCAST_IP) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_BROADCAST_IP);
				newOption.setData(DHCP_SERVER_BROADCAST_IP.getBytes());
				newOption.setLength((byte) 4);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER);
				newOption.setData(DHCP_SERVER_DHCP_SERVER_IP.getBytes());
				newOption.setLength((byte) 4);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME);
				newOption.setData(intToBytes(DHCP_SERVER_DEFAULT_LEASE_TIME_SECONDS));
				newOption.setLength((byte) 4);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_NTP_IP) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_NTP_IP);
				newOption.setData(DHCP_SERVER_NTP_IP_LIST);
				newOption.setLength((byte) DHCP_SERVER_NTP_IP_LIST.length);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME);
				newOption.setData(intToBytes(DHCP_SERVER_REBIND_TIME_SECONDS));
				newOption.setLength((byte) 4);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME);
				newOption.setData(intToBytes(DHCP_SERVER_RENEWAL_TIME_SECONDS));
				newOption.setLength((byte) 4);
				dhcpAckOptions.add(newOption);
			} else if (specificRequest.byteValue() == DHCP_REQ_PARAM_OPTION_CODE_IP_FORWARDING) {
				newOption = new DHCPOption();
				newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_IP_FORWARDING);
				newOption.setData(DHCP_SERVER_IP_FORWARDING);
				newOption.setLength((byte) 1);
				dhcpAckOptions.add(newOption);
			}else {
				log.debug("Setting specific request for ACK failed");
			}
		}
		newOption = new DHCPOption();
		newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_END);
		newOption.setLength((byte) 0);
		dhcpAckOptions.add(newOption);
		dhcpDHCPAck.setOptions(dhcpAckOptions);
		ethDHCPAck.setPayload(ipv4DHCPAck.setPayload(udpDHCPAck.setPayload(dhcpDHCPAck)));
		DHCPACKPacket.setInPort(OFPort.ANY);
		List<OFAction> actions = new ArrayList<OFAction>(1);
		actions.add(sw.getOFFactory().actions().output(inPort, 0xffFFffFF));
		DHCPACKPacket.setActions(actions);
		DHCPACKPacket.setData(ethDHCPAck.serialize());
		log.debug("Sending DHCP ACK");
		sw.write(DHCPACKPacket.build());
	}
	public void sendDHCPNack(IOFSwitch sw, OFPort inPort, MacAddress chaddr, IPv4Address giaddr, int xid) {
		OFPacketOut.Builder DHCPOfferPacket = sw.getOFFactory().buildPacketOut();
		DHCPOfferPacket.setBufferId(OFBufferId.NO_BUFFER);
		Ethernet ethDHCPOffer = new Ethernet();
		ethDHCPOffer.setSourceMACAddress(CONTROLLER_MAC);
		ethDHCPOffer.setDestinationMACAddress(chaddr);
		ethDHCPOffer.setEtherType(EthType.IPv4);
		IPv4 ipv4DHCPOffer = new IPv4();
		ipv4DHCPOffer.setDestinationAddress(BROADCAST_IP);
		ipv4DHCPOffer.setSourceAddress(CONTROLLER_IP);
		ipv4DHCPOffer.setProtocol(IpProtocol.UDP);
		ipv4DHCPOffer.setTtl((byte) 64);
		UDP udpDHCPOffer = new UDP();
		udpDHCPOffer.setDestinationPort(UDP.DHCP_CLIENT_PORT);
		udpDHCPOffer.setSourcePort(UDP.DHCP_SERVER_PORT);
		DHCP dhcpDHCPOffer = new DHCP();
		dhcpDHCPOffer.setOpCode(DHCP_OPCODE_REPLY);
		dhcpDHCPOffer.setHardwareType((byte) 1);
		dhcpDHCPOffer.setHardwareAddressLength((byte) 6);
		dhcpDHCPOffer.setHops((byte) 0);
		dhcpDHCPOffer.setTransactionId(xid);
		dhcpDHCPOffer.setSeconds((short) 0);
		dhcpDHCPOffer.setFlags((short) 0);
		dhcpDHCPOffer.setClientIPAddress(UNASSIGNED_IP);
		dhcpDHCPOffer.setYourIPAddress(UNASSIGNED_IP);
		dhcpDHCPOffer.setServerIPAddress(CONTROLLER_IP);
		dhcpDHCPOffer.setGatewayIPAddress(giaddr);
		dhcpDHCPOffer.setClientHardwareAddress(chaddr);
		List<DHCPOption> dhcpOfferOptions = new ArrayList<DHCPOption>();
		DHCPOption newOption;
		newOption = new DHCPOption();
		newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_MSG_TYPE);
		newOption.setData(DHCP_MSG_TYPE_NACK);
		newOption.setLength((byte) 1);
		dhcpOfferOptions.add(newOption);
		newOption = new DHCPOption();
		newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER);
		newOption.setData(DHCP_SERVER_DHCP_SERVER_IP.getBytes());
		newOption.setLength((byte) 4);
		dhcpOfferOptions.add(newOption);
		newOption = new DHCPOption();
		newOption.setCode(DHCP_REQ_PARAM_OPTION_CODE_END);
		newOption.setLength((byte) 0);
		dhcpOfferOptions.add(newOption);
		dhcpDHCPOffer.setOptions(dhcpOfferOptions);
		ethDHCPOffer.setPayload(ipv4DHCPOffer.setPayload(udpDHCPOffer.setPayload(dhcpDHCPOffer)));
		DHCPOfferPacket.setInPort(OFPort.ANY);
		List<OFAction> actions = new ArrayList<OFAction>(1);
		actions.add(sw.getOFFactory().actions().output(inPort, 0xffFFffFF));
		DHCPOfferPacket.setActions(actions);
		DHCPOfferPacket.setData(ethDHCPOffer.serialize());
		log.info("Sending DHCP NACK");
		sw.write(DHCPOfferPacket.build());
	}
	public ArrayList<Byte> getRequestedParameters(DHCP DHCPPayload, boolean isInform) {
		ArrayList<Byte> requestOrder = new ArrayList<Byte>();
		byte[] requests = DHCPPayload.getOption(DHCPOptionCode.OptionCode_RequestedParameters).getData();
		boolean requestedLeaseTime = false;
		boolean requestedRebindTime = false;
		boolean requestedRenewTime = false;
		for (byte specificRequest : requests) {
			if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_SN) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_SN);
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_ROUTER) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_ROUTER);
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_DN) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_DN);
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_DNS) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_DNS);
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME);
				requestedLeaseTime = true;
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER);
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_BROADCAST_IP) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_BROADCAST_IP);
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_NTP_IP) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_NTP_IP);
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME);
				requestedRebindTime = true;
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME);
				requestedRenewTime = true;
			} else if (specificRequest == DHCP_REQ_PARAM_OPTION_CODE_IP_FORWARDING) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_IP_FORWARDING);
				log.debug("requested IP FORWARDING");
			} else {
			}
		}
		if (!isInform) {
			if (!requestedLeaseTime) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_LEASE_TIME);
				log.debug("added option LEASE TIME");
			}
			if (!requestedRenewTime) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_RENEWAL_TIME);
				log.debug("added option RENEWAL TIME");
			}
			if (!requestedRebindTime) {
				requestOrder.add(DHCP_REQ_PARAM_OPTION_CODE_REBIND_TIME);
				log.debug("added option REBIND TIME");
			}
		}
		return requestOrder;
	}
	@Override
	public net.floodlightcontroller.core.IListener.Command receive(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		OFPacketIn pi = (OFPacketIn) msg;
		if (!theDHCPPool.hasAvailableAddresses()) {
			log.info("DHCP Pool is full! Consider increasing the pool size.");
			return Command.CONTINUE;
		}
		Ethernet eth = IFloodlightProviderService.bcStore.get(cntx,
				IFloodlightProviderService.CONTEXT_PI_PAYLOAD);
			log.debug("Got IPv4 Packet");
			IPv4 IPv4Payload = (IPv4) eth.getPayload();
			IPv4Address IPv4SrcAddr = IPv4Payload.getSourceAddress();
				log.debug("Got UDP Packet");
				UDP UDPPayload = (UDP) IPv4Payload.getPayload();
						|| UDPPayload.getDestinationPort().equals(UDP.DHCP_CLIENT_PORT))
						&& (UDPPayload.getSourcePort().equals(UDP.DHCP_SERVER_PORT)
						|| UDPPayload.getSourcePort().equals(UDP.DHCP_CLIENT_PORT)))
				{
					log.debug("Got DHCP Packet");
					DHCP DHCPPayload = (DHCP) UDPPayload.getPayload();
					OFPort inPort = (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT));
					int xid = 0;
					IPv4Address yiaddr = IPv4Address.NONE;
					IPv4Address giaddr = IPv4Address.NONE;
					MacAddress chaddr = null;
					IPv4Address desiredIPAddr = null;
					ArrayList<Byte> requestOrder = new ArrayList<Byte>();
					if (DHCPPayload.getOpCode() == DHCP_OPCODE_REQUEST) {
						if (Arrays.equals(DHCPPayload.getOption(DHCP.DHCPOptionCode.OptionCode_MessageType).getData(), DHCP_MSG_TYPE_DISCOVER)) {
							log.debug("DHCP DISCOVER Received");
							xid = DHCPPayload.getTransactionId();
							yiaddr = DHCPPayload.getYourIPAddress();
							giaddr = DHCPPayload.getGatewayIPAddress();
							chaddr = DHCPPayload.getClientHardwareAddress();
							List<DHCPOption> options = DHCPPayload.getOptions();
							for (DHCPOption option : options) {
								if (option.getCode() == DHCP_REQ_PARAM_OPTION_CODE_REQUESTED_IP) {
									desiredIPAddr = IPv4Address.of(option.getData());
									log.debug("Got requested IP");
								} else if (option.getCode() == DHCP_REQ_PARAM_OPTION_CODE_REQUESTED_PARAMTERS) {
									log.debug("Got requested param list");
									requestOrder = getRequestedParameters(DHCPPayload, false); 		
								}
							}
							synchronized (theDHCPPool) {
								if (!theDHCPPool.hasAvailableAddresses()) {
									log.info("DHCP Pool is full! Consider increasing the pool size.");
									log.info("Device with MAC " + chaddr.toString() + " was not granted an IP lease");
									return Command.CONTINUE;
								}
								DHCPBinding lease = theDHCPPool.getSpecificAvailableLease(desiredIPAddr, chaddr);
								if (lease != null) {
									log.debug("Checking new lease with specific IP");
									theDHCPPool.setDHCPbinding(lease, chaddr, DHCP_SERVER_HOLD_LEASE_TIME_SECONDS);
									yiaddr = lease.getIPv4Address();
									log.debug("Got new lease for " + yiaddr.toString());
								} else {
									log.debug("Checking new lease for any IP");
									lease = theDHCPPool.getAnyAvailableLease(chaddr);
									theDHCPPool.setDHCPbinding(lease, chaddr, DHCP_SERVER_HOLD_LEASE_TIME_SECONDS);
									yiaddr = lease.getIPv4Address();
									log.debug("Got new lease for " + yiaddr.toString());
								}
							}
							sendDHCPOffer(sw, inPort, chaddr, IPv4SrcAddr, yiaddr, giaddr, xid, requestOrder);
						else if (Arrays.equals(DHCPPayload.getOption(DHCP.DHCPOptionCode.OptionCode_MessageType).getData(), DHCP_MSG_TYPE_REQUEST)) {
							log.debug(": DHCP REQUEST received");
							IPv4SrcAddr = IPv4Payload.getSourceAddress();
							xid = DHCPPayload.getTransactionId();
							yiaddr = DHCPPayload.getYourIPAddress();
							giaddr = DHCPPayload.getGatewayIPAddress();
							chaddr = DHCPPayload.getClientHardwareAddress();
							List<DHCPOption> options = DHCPPayload.getOptions();
							for (DHCPOption option : options) {
								if (option.getCode() == DHCP_REQ_PARAM_OPTION_CODE_REQUESTED_IP) {
									desiredIPAddr = IPv4Address.of(option.getData());
									if (!desiredIPAddr.equals(theDHCPPool.getDHCPbindingFromMAC(chaddr).getIPv4Address())) {
										theDHCPPool.cancelLeaseOfMAC(chaddr);
										return Command.CONTINUE;
									}
								} else if (option.getCode() == DHCP_REQ_PARAM_OPTION_CODE_DHCP_SERVER) {
									if (!IPv4Address.of(option.getData()).equals(DHCP_SERVER_DHCP_SERVER_IP)) {
										theDHCPPool.cancelLeaseOfMAC(chaddr);
										return Command.CONTINUE;
									}
								} else if (option.getCode() == DHCP_REQ_PARAM_OPTION_CODE_REQUESTED_PARAMTERS) {
									requestOrder = getRequestedParameters(DHCPPayload, false);
								}
							}
							boolean sendACK = true;
							synchronized (theDHCPPool) {
								if (!theDHCPPool.hasAvailableAddresses()) {
									log.info("DHCP Pool is full! Consider increasing the pool size.");
									log.info("Device with MAC " + chaddr.toString() + " was not granted an IP lease");
									return Command.CONTINUE;
								}
								DHCPBinding lease;
								if (desiredIPAddr != null) {
									lease = theDHCPPool.getDHCPbindingFromIPv4(desiredIPAddr);
								} else {
									lease = theDHCPPool.getAnyAvailableLease(chaddr);
								}
								if (lease == null) {
									log.info("The IP " + desiredIPAddr.toString() + " is not in the range " 
											+ DHCP_SERVER_IP_START.toString() + " to " + DHCP_SERVER_IP_STOP.toString());
									log.info("Device with MAC " + chaddr.toString() + " was not granted an IP lease");
									sendACK = false;
								} else if (!lease.getMACAddress().equals(chaddr) && lease.isActiveLease()) {
									log.debug("Tried to REQUEST an IP that is currently assigned to another MAC");
									log.debug("Device with MAC " + chaddr.toString() + " was not granted an IP lease");
									sendACK = false;
								} else if (lease.getMACAddress().equals(chaddr) && lease.isActiveLease()) {
									log.debug("Renewing lease for MAC " + chaddr.toString());
									theDHCPPool.renewLease(lease.getIPv4Address(), DHCP_SERVER_DEFAULT_LEASE_TIME_SECONDS);
									yiaddr = lease.getIPv4Address();
									log.debug("Finalized renewed lease for " + yiaddr.toString());
								} else if (!lease.isActiveLease()){
									log.debug("Assigning new lease for MAC " + chaddr.toString());
									theDHCPPool.setDHCPbinding(lease, chaddr, DHCP_SERVER_DEFAULT_LEASE_TIME_SECONDS);
									yiaddr = lease.getIPv4Address();
									log.debug("Finalized new lease for " + yiaddr.toString());
								} else {
									log.debug("Don't know how we got here");
									return Command.CONTINUE;
								}
							}
							if (sendACK) {
								sendDHCPAck(sw, inPort, chaddr, IPv4SrcAddr, yiaddr, giaddr, xid, requestOrder);							
							} else {
								sendDHCPNack(sw, inPort, chaddr, giaddr, xid);
							}
						else if (Arrays.equals(DHCPPayload.getOption(DHCP.DHCPOptionCode.OptionCode_MessageType).getData(), DHCP_MSG_TYPE_RELEASE)) {
							if (DHCPPayload.getServerIPAddress() != CONTROLLER_IP) {
								log.info("DHCP RELEASE message not for our DHCP server");
							} else {
								log.debug("Got DHCP RELEASE. Cancelling remaining time on DHCP lease");
								synchronized(theDHCPPool) {
									if (theDHCPPool.cancelLeaseOfMAC(DHCPPayload.getClientHardwareAddress())) {
										log.info("Cancelled DHCP lease of " + DHCPPayload.getClientHardwareAddress().toString());
										log.info("IP " + theDHCPPool.getDHCPbindingFromMAC(DHCPPayload.getClientHardwareAddress()).getIPv4Address().toString()
												+ " is now available in the DHCP address pool");
									} else {
										log.debug("Lease of " + DHCPPayload.getClientHardwareAddress().toString()
												+ " was already inactive");
									}
								}
							}
						else if (Arrays.equals(DHCPPayload.getOption(DHCP.DHCPOptionCode.OptionCode_MessageType).getData(), DHCP_MSG_TYPE_DECLINE)) {
							log.debug("Got DHCP DECLINE. Cancelling HOLD time on DHCP lease");
							synchronized(theDHCPPool) {
								if (theDHCPPool.cancelLeaseOfMAC(DHCPPayload.getClientHardwareAddress())) {
									log.info("Cancelled DHCP lease of " + DHCPPayload.getClientHardwareAddress().toString());
									log.info("IP " + theDHCPPool.getDHCPbindingFromMAC(DHCPPayload.getClientHardwareAddress()).getIPv4Address().toString()
											+ " is now available in the DHCP address pool");
								} else {
									log.info("HOLD Lease of " + DHCPPayload.getClientHardwareAddress().toString()
											+ " has already expired");
								}
							}
						else if (Arrays.equals(DHCPPayload.getOption(DHCP.DHCPOptionCode.OptionCode_MessageType).getData(), DHCP_MSG_TYPE_INFORM)) {
							log.debug("Got DHCP INFORM. Retreiving requested parameters from message");
							IPv4SrcAddr = IPv4Payload.getSourceAddress();
							xid = DHCPPayload.getTransactionId();
							yiaddr = DHCPPayload.getYourIPAddress();
							giaddr = DHCPPayload.getGatewayIPAddress();
							chaddr = DHCPPayload.getClientHardwareAddress();
							requestOrder = getRequestedParameters(DHCPPayload, true);
							sendDHCPAck(sw, inPort, chaddr, IPv4SrcAddr, yiaddr, giaddr, xid, requestOrder);							
					else if (DHCPPayload.getOpCode() == DHCP_OPCODE_REPLY) {
						log.debug("Got an OFFER/ACK (REPLY)...this shouldn't happen unless there's another DHCP Server somewhere");
					} else {
						log.debug("Got DHCP packet, but not a known DHCP packet opcode");
					}
		return Command.CONTINUE;
	class DHCPLeasePolice implements Runnable {
		@Override
		public void run() {
			log.info("Cleaning any expired DHCP leases...");
			ArrayList<DHCPBinding> newAvailableBindings;
			synchronized(theDHCPPool) {
				newAvailableBindings = theDHCPPool.cleanExpiredLeases();
			}
			for (DHCPBinding binding : newAvailableBindings) {
				log.info("MAC " + binding.getMACAddress().toString() + " has expired");
				log.info("Lease now available for IP " + binding.getIPv4Address().toString());
			}
		}
