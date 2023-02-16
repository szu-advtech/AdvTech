package net.floodlightcontroller.util;
import java.util.ArrayList;
import java.util.List;
import net.floodlightcontroller.core.IOFSwitch;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPacketOut;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFPort;
public class OFMessageUtils {
	private OFMessageUtils() {};
	public static boolean equalsIgnoreXid(OFMessage a, OFMessage b) {
		OFMessage.Builder mb = b.createBuilder().setXid(a.getXid());
		return a.equals(mb.build());
	}
	public static void writePacketOutForPacketIn(IOFSwitch sw,
			OFPacketIn packetInMessage, OFPort egressPort) {
		OFPacketOut.Builder pob = sw.getOFFactory().buildPacketOut();
		pob.setBufferId(packetInMessage.getBufferId());
		pob.setInPort(packetInMessage.getVersion().compareTo(OFVersion.OF_12) < 0 ? packetInMessage
				.getInPort() : packetInMessage.getMatch().get(
				MatchField.IN_PORT));
		List<OFAction> actions = new ArrayList<OFAction>(1);
		actions.add(sw.getOFFactory().actions().buildOutput()
				.setPort(egressPort).setMaxLen(0xffFFffFF).build());
		pob.setActions(actions);
		if (packetInMessage.getBufferId() == OFBufferId.NO_BUFFER) {
			byte[] packetData = packetInMessage.getData();
			pob.setData(packetData);
		}
		sw.write(pob.build());
	}
}
