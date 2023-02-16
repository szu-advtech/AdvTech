package net.floodlightcontroller.savi.action;
import java.util.ArrayList;
import java.util.List;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.packet.Ethernet;
public class PacketOutAction extends Action {
	Ethernet eth;
	DatapathId switchId;
	OFPort inPort;
	List<OFPort> outPorts; 
	public PacketOutAction() {
		this.type = ActionType.PACKET_OUT;
		this.inPort = null;
		this.switchId = null;
		this.eth = null;
		this.outPorts = new ArrayList<>();
	}
	public PacketOutAction(Ethernet eth, DatapathId switchId, OFPort inPort, List<OFPort> outPorts){
		this.type = ActionType.PACKET_OUT;
		this.eth = eth;
		this.inPort = inPort;
		this.switchId = switchId;
		this.outPorts = outPorts;
	}
	public Ethernet getEthernet() {
		return eth;
	}
	public void setEthernet(Ethernet eth) {
		this.eth = eth;
	}
	public DatapathId getSwitchId() {
		return switchId;
	}
	public void setSwitchId(DatapathId switchId) {
		this.switchId = switchId;
	}
	public OFPort getInPort() {
		return inPort;
	}
	public void setInPort(OFPort inPort) {
		this.inPort = inPort;
	}
	public List<OFPort> getOutPorts() {
		return outPorts;
	}
	public void setOutPorts(List<OFPort> outPorts) {
		this.outPorts = outPorts;
	}
	public void addOutPort(OFPort outPort) {
		this.outPorts.add(outPort);
	}
}
