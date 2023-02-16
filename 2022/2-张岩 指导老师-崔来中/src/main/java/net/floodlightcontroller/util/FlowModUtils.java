package net.floodlightcontroller.util;
import java.util.Collections;
import java.util.List;
import net.floodlightcontroller.core.IOFSwitch;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFlowAdd;
import org.projectfloodlight.openflow.protocol.OFFlowDelete;
import org.projectfloodlight.openflow.protocol.OFFlowDeleteStrict;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.protocol.OFFlowModify;
import org.projectfloodlight.openflow.protocol.OFFlowModifyStrict;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
public class FlowModUtils {
	public static final int INFINITE_TIMEOUT = 0;
	public static final int PRIORITY_MAX = 32768;
	public static final int PRIORITY_VERY_HIGH = 28672;
	public static final int PRIORITY_HIGH = 24576;
	public static final int PRIORITY_MED_HIGH = 20480;
	public static final int PRIORITY_MED = 16384;
	public static final int PRIORITY_MED_LOW = 12288;
	public static final int PRIORITY_LOW = 8192;
	public static final int PRIORITY_VERY_LOW = 4096;
	public static final int PRIORITY_MIN = 0;
	public static OFFlowAdd toFlowAdd(OFFlowMod fm) {
		OFVersion version = fm.getVersion();
		OFFlowAdd.Builder b = OFFactories.getFactory(version).buildFlowAdd();
		if (b.getVersion().compareTo(OFVersion.OF_10) == 0) {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setXid(fm.getXid())
					.build();
		} else {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setTableId(fm.getTableId())
					.setXid(fm.getXid())
					.build();
		}
	}
	public static OFFlowDelete toFlowDelete(OFFlowMod fm) {
		OFVersion version = fm.getVersion();
		OFFlowDelete.Builder b = OFFactories.getFactory(version).buildFlowDelete();
		if (b.getVersion().compareTo(OFVersion.OF_10) == 0) {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setXid(fm.getXid())
					.build();
		} else {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setTableId(fm.getTableId())
					.setXid(fm.getXid())
					.build();
		}
	}
	public static OFFlowDeleteStrict toFlowDeleteStrict(OFFlowMod fm) {
		OFVersion version = fm.getVersion();
		OFFlowDeleteStrict.Builder b = OFFactories.getFactory(version).buildFlowDeleteStrict();
		if (b.getVersion().compareTo(OFVersion.OF_10) == 0) {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setXid(fm.getXid())
					.build();
		} else {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setTableId(fm.getTableId())
					.setXid(fm.getXid())
					.build();
		}
	}
	public static OFFlowModify toFlowModify(OFFlowMod fm) {
		OFVersion version = fm.getVersion();
		OFFlowModify.Builder b = OFFactories.getFactory(version).buildFlowModify();
		if (b.getVersion().compareTo(OFVersion.OF_10) == 0) {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setXid(fm.getXid())
					.build();
		} else {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setTableId(fm.getTableId())
					.setXid(fm.getXid())
					.build();
		}
	}
	public static OFFlowModifyStrict toFlowModifyStrict(OFFlowMod fm) {
		OFVersion version = fm.getVersion();
		OFFlowModifyStrict.Builder b = OFFactories.getFactory(version).buildFlowModifyStrict();
		if (b.getVersion().compareTo(OFVersion.OF_10) == 0) {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setXid(fm.getXid())
					.build();
		} else {
			return b.setActions(fm.getActions())
					.setBufferId(fm.getBufferId())
					.setCookie(fm.getCookie())
					.setFlags(fm.getFlags())
					.setHardTimeout(fm.getHardTimeout())
					.setIdleTimeout(fm.getIdleTimeout())
					.setMatch(fm.getMatch())
					.setOutPort(fm.getOutPort())
					.setPriority(fm.getPriority())
					.setTableId(fm.getTableId())
					.setXid(fm.getXid())
					.build();
		}
	}
	public static void setActions(OFFlowMod.Builder fmb,
			List<OFAction> actions, IOFSwitch sw) {
		if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_11) >= 0) {
			fmb.setInstructions(Collections.singletonList((OFInstruction) sw
					.getOFFactory().instructions().applyActions(actions)));
		} else {
			fmb.setActions(actions);
		}
	}
}
