package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.packet.Ethernet;
public class FloodAction extends Action {
	Ethernet eth;
	protected DatapathId switchId;
	protected OFPort	inPort;
	public FloodAction() {
		this.type = ActionType.FLOOD;
		this.switchId = null;
		this.inPort = null;
	}
	public FloodAction(DatapathId switchId,OFPort inPort, Ethernet eth) {
		this.type = ActionType.FLOOD;
		this.inPort	= inPort;
		this.switchId = switchId;
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
	public Ethernet getEthernet() {
		return eth;
	}
	public void setEthernet(Ethernet eth) {
		this.eth = eth;
	}
}
