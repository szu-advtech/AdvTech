package net.floodlightcontroller.core.internal;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nonnull;
import io.netty.util.Timer;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IOFConnection;
import net.floodlightcontroller.core.IOFConnectionBackend;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitch.SwitchStatus;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.PortChangeEvent;
import net.floodlightcontroller.core.SwitchDescription;
import net.floodlightcontroller.core.internal.OFSwitchAppHandshakePlugin.PluginResultType;
import org.projectfloodlight.openflow.protocol.OFActionType;
import org.projectfloodlight.openflow.protocol.OFBadRequestCode;
import org.projectfloodlight.openflow.protocol.OFBarrierReply;
import org.projectfloodlight.openflow.protocol.OFBarrierRequest;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.protocol.OFDescStatsReply;
import org.projectfloodlight.openflow.protocol.OFDescStatsRequest;
import org.projectfloodlight.openflow.protocol.OFErrorMsg;
import org.projectfloodlight.openflow.protocol.OFErrorType;
import org.projectfloodlight.openflow.protocol.OFExperimenter;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
import org.projectfloodlight.openflow.protocol.OFFlowAdd;
import org.projectfloodlight.openflow.protocol.OFFlowDelete;
import org.projectfloodlight.openflow.protocol.OFFlowDeleteStrict;
import org.projectfloodlight.openflow.protocol.OFFlowModFailedCode;
import org.projectfloodlight.openflow.protocol.OFFlowRemoved;
import org.projectfloodlight.openflow.protocol.OFGetConfigReply;
import org.projectfloodlight.openflow.protocol.OFGetConfigRequest;
import org.projectfloodlight.openflow.protocol.OFGroupDelete;
import org.projectfloodlight.openflow.protocol.OFGroupType;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFNiciraControllerRole;
import org.projectfloodlight.openflow.protocol.OFNiciraControllerRoleReply;
import org.projectfloodlight.openflow.protocol.OFNiciraControllerRoleRequest;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPortDescStatsReply;
import org.projectfloodlight.openflow.protocol.OFPortStatus;
import org.projectfloodlight.openflow.protocol.OFQueueGetConfigReply;
import org.projectfloodlight.openflow.protocol.OFRoleReply;
import org.projectfloodlight.openflow.protocol.OFRoleRequest;
import org.projectfloodlight.openflow.protocol.OFSetConfig;
import org.projectfloodlight.openflow.protocol.OFStatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsReplyFlags;
import org.projectfloodlight.openflow.protocol.OFStatsRequestFlags;
import org.projectfloodlight.openflow.protocol.OFStatsType;
import org.projectfloodlight.openflow.protocol.OFTableFeaturesStatsReply;
import org.projectfloodlight.openflow.protocol.OFTableFeaturesStatsRequest;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.actionid.OFActionId;
import org.projectfloodlight.openflow.protocol.errormsg.OFBadRequestErrorMsg;
import org.projectfloodlight.openflow.protocol.errormsg.OFFlowModFailedErrorMsg;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFAuxId;
import org.projectfloodlight.openflow.types.OFGroup;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
public class OFSwitchHandshakeHandler implements IOFConnectionListener {
	private static final Logger log = LoggerFactory.getLogger(OFSwitchHandshakeHandler.class);
	private final IOFSwitchManager switchManager;
	private final RoleManager roleManager;
	private final IOFConnectionBackend mainConnection;
	private final SwitchManagerCounters switchManagerCounters;
	private IOFSwitchBackend sw;
	private final Map<OFAuxId, IOFConnectionBackend> auxConnections;
	private volatile OFSwitchHandshakeState state;
	private RoleChanger roleChanger;
	private OFFactory factory = OFFactories.getFactory(OFVersion.OF_14);
	private final OFFeaturesReply featuresReply;
	private final Timer timer;
	private volatile OFControllerRole initialRole = null;
	private final ArrayList<OFPortStatus> pendingPortStatusMsg;
	private long handshakeTransactionIds = 0x00FFFFFFFFL;
	private final long MAX_ASSERT_TIME_INTERVAL_NS = TimeUnit.SECONDS.toNanos(120);
	private final long DEFAULT_ROLE_TIMEOUT_NS = TimeUnit.SECONDS.toNanos(10);
	protected OFPortDescStatsReply portDescStats;
	private enum RoleRecvStatus {
		RECEIVED_REPLY,
		UNSUPPORTED,
		NO_REPLY;
	}
	private class RoleChanger {
		private volatile boolean requestPending;
		private long pendingXid;
		private OFControllerRole pendingRole;
		private long roleSubmitTimeNs;
		private final long roleTimeoutNs;
		private long lastAssertTimeNs;
		private long assertTimeIntervalNs = TimeUnit.SECONDS.toNanos(1);
		public RoleChanger(long roleTimeoutNs) {
			this.roleTimeoutNs = roleTimeoutNs;
			this.requestPending = false;
			this.pendingXid = -1;
			this.pendingRole = null;
		}
		private long sendNiciraRoleRequest(OFControllerRole role, long xid){
			if(factory.getVersion().compareTo(OFVersion.OF_12) < 0) {
				OFNiciraControllerRoleRequest.Builder builder =
						factory.buildNiciraControllerRoleRequest();
				xid = xid <= 0 ? factory.nextXid() : xid;
				builder.setXid(xid);
				OFNiciraControllerRole niciraRole = NiciraRoleUtils.ofRoleToNiciraRole(role);
				builder.setRole(niciraRole);
				OFNiciraControllerRoleRequest roleRequest = builder.build();
				mainConnection.write(roleRequest);
			} else {
				OFRoleRequest roleRequest = factory.buildRoleRequest()
						.setGenerationId(U64.of(0))
						.setXid(xid <= 0 ? factory.nextXid() : xid)
						.setRole(role)
						.build();
				xid = roleRequest.getXid();
				mainConnection.write(roleRequest);
			}
			return xid;
		}
		synchronized void sendRoleRequestIfNotPending(OFControllerRole role, long xid)
				throws IOException {
			long now = System.nanoTime();
			if (now - lastAssertTimeNs < assertTimeIntervalNs) {
				return;
			}
			lastAssertTimeNs = now;
				assertTimeIntervalNs <<= 1;
			} else if (role == OFControllerRole.ROLE_MASTER){
				log.warn("Reasserting master role on switch {}, " +
						"likely a switch config error with multiple masters",
						role, sw);
			}
			if (!requestPending)
				sendRoleRequest(role, xid);
			else
				switchManagerCounters.roleNotResentBecauseRolePending.increment();
		}
		synchronized void sendRoleRequest(OFControllerRole role, long xid) throws IOException {
			Boolean supportsNxRole = (Boolean)
					sw.getAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE);
			if ((supportsNxRole != null) && !supportsNxRole) {
				setSwitchRole(role, RoleRecvStatus.UNSUPPORTED);
			} else {
				pendingXid = sendNiciraRoleRequest(role, xid);
				pendingRole = role;
				this.roleSubmitTimeNs = System.nanoTime();
				requestPending = true;
			}
		}
		synchronized void deliverRoleReply(long xid, OFControllerRole role) {
			log.debug("DELIVERING ROLE REPLY {}", role.toString());
			if (!requestPending) {
				String msg = String.format("Switch: [%s], State: [%s], "
						+ "received unexpected RoleReply[%s]. "
						+ "No roles are pending",
						OFSwitchHandshakeHandler.this.getSwitchInfoString(),
						OFSwitchHandshakeHandler.this.state.toString(),
						role);
				throw new SwitchStateException(msg);
			}
			if (pendingXid == xid && pendingRole == role) {
				log.debug("[{}] Received role reply message setting role to {}",
						getDpid(), role);
				switchManagerCounters.roleReplyReceived.increment();
				setSwitchRole(role, RoleRecvStatus.RECEIVED_REPLY);
			} else {
				log.debug("[{}] Received stale or unexpected role reply " +
						"{}, xid={}. Ignoring. " +
						"Waiting for {}, xid={}",
						new Object[] { getDpid(), role, xid,
						pendingRole, pendingXid });
			}
		}
		synchronized boolean deliverError(OFErrorMsg error) {
			if (!requestPending)
				return false;
			if (pendingXid == error.getXid()) {
				if (error.getErrType() == OFErrorType.BAD_REQUEST) {
					switchManagerCounters.roleReplyErrorUnsupported.increment();
					setSwitchRole(pendingRole, RoleRecvStatus.UNSUPPORTED);
				} else {
					String msg = String.format("Switch: [%s], State: [%s], "
							+ "Unexpected error %s in respone to our "
							+ "role request for %s.",
							OFSwitchHandshakeHandler.this.getSwitchInfoString(),
							OFSwitchHandshakeHandler.this.state.toString(),
							error.toString(),
							pendingRole);
					throw new SwitchStateException(msg);
				}
				return true;
			}
			return false;
		}
		void checkTimeout() {
			if (!requestPending)
				return;
			synchronized(this) {
				if (!requestPending)
					return;
				long now = System.nanoTime();
				if (now - this.roleSubmitTimeNs > roleTimeoutNs) {
					switchManagerCounters.roleReplyTimeout.increment();
					setSwitchRole(pendingRole, RoleRecvStatus.NO_REPLY);
				}
			}
		}
		synchronized private void setSwitchRole(OFControllerRole role, RoleRecvStatus status) {
			requestPending = false;
			if (status == RoleRecvStatus.RECEIVED_REPLY)
				sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, true);
			else
				sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, false);
			sw.setControllerRole(role);
			if (role != OFControllerRole.ROLE_SLAVE) {
				OFSwitchHandshakeHandler.this.setState(new MasterState());
			} else {
				if (status != RoleRecvStatus.RECEIVED_REPLY) {
					if (log.isDebugEnabled()) {
						log.debug("Disconnecting switch {}. Doesn't support role"
								+ "({}) request and controller is now SLAVE",
								getSwitchInfoString(), status);
					}
					sw.disconnect();
				} else {
					OFSwitchHandshakeHandler.this.setState(new SlaveState());
				}
			}
		}
	}
	private void clearAllTables() {
		if (this.sw.getOFFactory().getVersion().compareTo(OFVersion.OF_10) == 0) {
			OFFlowDelete deleteFlows = this.factory.buildFlowDelete()
					.build();
			this.sw.write(deleteFlows);
			OFFlowDelete deleteFlows = this.factory.buildFlowDelete()
					.setTableId(TableId.ALL)
					.build();
			this.sw.write(deleteFlows);
			OFGroupDelete delgroup = this.sw.getOFFactory().buildGroupDelete()
					.setGroup(OFGroup.ALL)
					.setGroupType(OFGroupType.ALL)
					.build();
			this.sw.write(delgroup);
			delgroup.createBuilder()
			.setGroupType(OFGroupType.FF)
			.build();
			this.sw.write(delgroup);
			delgroup.createBuilder()
			.setGroupType(OFGroupType.INDIRECT)
			.build();
			this.sw.write(delgroup);
			delgroup.createBuilder()
			.setGroupType(OFGroupType.SELECT)
			.build();
			this.sw.write(delgroup);
			OFBarrierRequest barrier = factory.buildBarrierRequest()
					.setXid(handshakeTransactionIds--)
					.build();
			sw.write(barrier);
		}
	}
	private void addDefaultFlows() {
		if (this.sw.getOFFactory().getVersion().compareTo(OFVersion.OF_13) >= 0) {
			OFFlowDeleteStrict deleteFlow = this.factory.buildFlowDeleteStrict()
					.setTableId(TableId.ALL)
					.setOutPort(OFPort.CONTROLLER)
					.build();
			this.sw.write(deleteFlow);
			ArrayList<OFAction> actions = new ArrayList<OFAction>(1);
			actions.add(factory.actions().output(OFPort.CONTROLLER, 0xffFFffFF));
			ArrayList<OFMessage> flows = new ArrayList<OFMessage>();
			if (!this.sw.getTables().isEmpty()) {
				short missCount = 0;
				for (TableId tid : this.sw.getTables()) {
					TableFeatures tf = this.sw.getTableFeatures(tid);
					if (tf != null && (missCount < this.sw.getMaxTableForTableMissFlow().getValue())) {
						for (OFActionId aid : tf.getPropApplyActionsMiss().getActionIds()) {
								OFFlowAdd defaultFlow = this.factory.buildFlowAdd()
										.setTableId(tid)
										.setPriority(0)
										.setInstructions(Collections.singletonList((OFInstruction) this.factory.instructions().buildApplyActions().setActions(actions).build()))
										.build();
								flows.add(defaultFlow);
							}
						}
					}
					missCount++;
				}
				short missCount = 0;
				for (short tid = 0; tid < this.sw.getNumTables(); tid++, missCount++) {
						OFFlowAdd defaultFlow = this.factory.buildFlowAdd()
								.setTableId(TableId.of(tid))
								.setPriority(0)
								.setActions(actions)
								.build();
						flows.add(defaultFlow);
					}
				}
			}
			this.sw.write(flows);
		}
	}
	public abstract class OFSwitchHandshakeState {
		void processOFBarrierReply(OFBarrierReply m) {
		}
		void processOFError(OFErrorMsg m) {
			logErrorDisconnect(m);
		}
		void processOFFlowRemoved(OFFlowRemoved m) {
			unhandledMessageReceived(m);
		}
		void processOFGetConfigReply(OFGetConfigReply m) {
			illegalMessageReceived(m);
		}
		void processOFPacketIn(OFPacketIn m) {
			unhandledMessageReceived(m);
		}
		void processOFPortStatus(OFPortStatus m) {
			pendingPortStatusMsg.add(m);
		}
		void processOFQueueGetConfigReply(OFQueueGetConfigReply m) {
			unhandledMessageReceived(m);
		}
		void processOFStatsReply(OFStatsReply m) {
			switch(m.getStatsType()) {
			case PORT_DESC:
				processPortDescStatsReply((OFPortDescStatsReply) m);
				break;
			default:
				unhandledMessageReceived(m);
			}
		}
		void processOFExperimenter(OFExperimenter m) {
			unhandledMessageReceived(m);
		}
		void processPortDescStatsReply(OFPortDescStatsReply m) {
			unhandledMessageReceived(m);
		}
		void processOFRoleReply(OFRoleReply m) {
			unhandledMessageReceived(m);
		}
		void processOFRoleRequest(OFRoleRequest m) {
			unhandledMessageWritten(m);
		}
		void processOFNiciraControllerRoleRequest(OFNiciraControllerRoleRequest m) {
			unhandledMessageWritten(m);
		}
		private final boolean handshakeComplete;
		OFSwitchHandshakeState(boolean handshakeComplete) {
			this.handshakeComplete = handshakeComplete;
		}
		void logState() {
			if(log.isDebugEnabled())
				log.debug("[{}] - Switch Handshake - enter state {}", mainConnection.getDatapathId(), this.getClass().getSimpleName());
		}
		void enterState(){
		}
		public boolean isHandshakeComplete() {
			return handshakeComplete;
		}
		public void auxConnectionOpened(IOFConnectionBackend connection) {
			log.debug("[{}] - Switch Handshake - unhandled aux connection event",
					getDpid());
		}
		protected String getSwitchStateMessage(OFMessage m,
				String details) {
			return String.format("Switch: [%s], State: [%s], received: [%s]"
					+ ", details: %s",
					getSwitchInfoString(),
					this.toString(),
					m.getType().toString(),
					details);
		}
		protected void illegalMessageReceived(OFMessage m) {
			String msg = getSwitchStateMessage(m,
					"Switch should never send this message in the current state");
			throw new SwitchStateException(msg);
		}
		protected void unhandledMessageReceived(OFMessage m) {
			switchManagerCounters.unhandledMessage.increment();
			if (log.isDebugEnabled()) {
				String msg = getSwitchStateMessage(m,
						"Ignoring unexpected message");
				log.debug(msg);
			}
		}
		protected void unhandledMessageWritten(OFMessage m) {
			switchManagerCounters.unhandledMessage.increment();
			if (log.isDebugEnabled()) {
				String msg = getSwitchStateMessage(m,
						"Ignoring unexpected written message");
				log.debug(msg);
			}
		}
		protected void logError(OFErrorMsg error) {
			log.error("{} from switch {} in state {}",
					new Object[] {
					error.toString(),
					getSwitchInfoString(),
					this.toString()});
		}
		protected void logErrorDisconnect(OFErrorMsg error) {
			logError(error);
			mainConnection.disconnect();
		}
		protected OFControllerRole extractNiciraRoleReply(OFMessage vendorMessage) {
			if (!(vendorMessage instanceof OFNiciraControllerRoleReply))
				return null;
			OFNiciraControllerRoleReply roleReply =
					(OFNiciraControllerRoleReply) vendorMessage;
			return NiciraRoleUtils.niciraToOFRole(roleReply);
		}
		protected void handlePortStatusMessage(OFPortStatus m, boolean doNotify) {
			if (sw == null) {
				String msg = getSwitchStateMessage(m, "State machine error: switch is null. Should never happen");
				throw new SwitchStateException(msg);
			}
			Collection<PortChangeEvent> changes = sw.processOFPortStatus(m);
			if (doNotify) {
				for (PortChangeEvent ev: changes)
					switchManager.notifyPortChanged(sw, ev.port, ev.type);
			}
		}
		protected void handleTableFeaturesMessage(List<OFTableFeaturesStatsReply> replies, boolean doNotify) {
			if (sw == null) {
				String msg = getSwitchStateMessage(!replies.isEmpty() ? replies.get(0) : null, "State machine error: switch is null. Should never happen");
				throw new SwitchStateException(msg);
			}
			sw.processOFTableFeatures(replies);
		}
		void processOFMessage(OFMessage m) {
			roleChanger.checkTimeout();
			switch(m.getType()) {
			case BARRIER_REPLY:
				processOFBarrierReply((OFBarrierReply) m);
				break;
			case ERROR:
				processOFError((OFErrorMsg) m);
				break;
			case FLOW_REMOVED:
				processOFFlowRemoved((OFFlowRemoved) m);
				break;
			case GET_CONFIG_REPLY:
				processOFGetConfigReply((OFGetConfigReply) m);
				break;
			case PACKET_IN:
				processOFPacketIn((OFPacketIn) m);
				break;
			case PORT_STATUS:
				processOFPortStatus((OFPortStatus) m);
				break;
			case QUEUE_GET_CONFIG_REPLY:
				processOFQueueGetConfigReply((OFQueueGetConfigReply) m);
				break;
			case STATS_REPLY:
				processOFStatsReply((OFStatsReply) m);
				break;
			case ROLE_REPLY:
				processOFRoleReply((OFRoleReply) m);
				break;
			case EXPERIMENTER:
				processOFExperimenter((OFExperimenter) m);
				break;
			default:
				illegalMessageReceived(m);
				break;
			}
		}
		void processWrittenOFMessage(OFMessage m) {
			switch(m.getType()) {
			case ROLE_REQUEST:
				processOFRoleRequest((OFRoleRequest) m);
				break;
			case EXPERIMENTER:
				if (m instanceof OFNiciraControllerRoleRequest) {
					processOFNiciraControllerRoleRequest((OFNiciraControllerRoleRequest) m);
				}
				break;
			default:
				break;
			}
		}
	}
	public class InitState extends OFSwitchHandshakeState {
		InitState() {
			super(false);
		}
		@Override
		public void logState() {
			log.debug("[{}] - Switch Handshake - Initiating from {}",
					getDpid(), mainConnection.getRemoteInetAddress());
		}
	}
	public class WaitPortDescStatsReplyState extends OFSwitchHandshakeState {
		WaitPortDescStatsReplyState() {
			super(false);
		}
		@Override
		void enterState(){
			sendPortDescRequest();
		}
		@Override
		void processPortDescStatsReply(OFPortDescStatsReply  m) {
			portDescStats = m;
			setState(new WaitConfigReplyState());
		}
		@Override
		void processOFExperimenter(OFExperimenter m) {
			unhandledMessageReceived(m);
		}
	}
	public class WaitConfigReplyState extends OFSwitchHandshakeState {		
		WaitConfigReplyState() {
			super(false);
		}
		@Override
		void processOFGetConfigReply(OFGetConfigReply m) {
			if (m.getMissSendLen() == 0xffff) {
				log.trace("Config Reply from switch {} confirms "
						+ "miss length set to 0xffff",
						getSwitchInfoString());
			} else {
				log.warn("Config Reply from switch {} has"
						+ "miss length set to {}",
						getSwitchInfoString(),
						m.getMissSendLen());
			}
			setState(new WaitDescriptionStatReplyState());
		}
		@Override
		void processOFStatsReply(OFStatsReply  m) {
			illegalMessageReceived(m);
		}
		@Override
		void processOFError(OFErrorMsg m) {
			if (m.getErrType() == OFErrorType.BAD_REQUEST &&
					((OFBadRequestErrorMsg) m).getCode() == OFBadRequestCode.BAD_TYPE &&
					((OFBadRequestErrorMsg) m).getData().getParsedMessage().get() instanceof OFBarrierRequest) {
				log.warn("Switch does not support Barrier Request messages. Could be an HP ProCurve.");
			} else {
				logErrorDisconnect(m);
			}
		} 
		@Override
		void enterState() {
			sendHandshakeSetConfig();
		}
	}
	public class WaitDescriptionStatReplyState extends OFSwitchHandshakeState{
		long timestamp;
		WaitDescriptionStatReplyState() {
			super(false);
		}
		@Override
		void processOFStatsReply(OFStatsReply m) {
			if (m.getStatsType() != OFStatsType.DESC) {
				illegalMessageReceived(m);
				return;
			}
			OFDescStatsReply descStatsReply = (OFDescStatsReply) m;
			SwitchDescription description = new SwitchDescription(descStatsReply);
			sw = switchManager.getOFSwitchInstance(mainConnection, description, factory, featuresReply.getDatapathId());
			sw.setFeaturesReply(featuresReply);
			if (portDescStats != null) {
				sw.setPortDescStats(portDescStats);
			}
			switchManager.switchAdded(sw);
			handlePendingPortStatusMessages(description);
			setState(new WaitTableFeaturesReplyState());
		}
		void handlePendingPortStatusMessages(SwitchDescription description){
			for (OFPortStatus ps: pendingPortStatusMsg) {
				handlePortStatusMessage(ps, false);
			}
			pendingPortStatusMsg.clear();
			log.info("Switch {} bound to class {}, description {}", new Object[] { sw, sw.getClass(), description });
		}
		@Override
		void enterState() {
			sendHandshakeDescriptionStatsRequest();
		}
	}
	public class WaitTableFeaturesReplyState extends OFSwitchHandshakeState {
		private ArrayList<OFTableFeaturesStatsReply> replies;
		WaitTableFeaturesReplyState() {
			super(false);
			replies = new ArrayList<OFTableFeaturesStatsReply>();
		}
		@Override
		void processOFStatsReply(OFStatsReply m) {
			if (m.getStatsType() == OFStatsType.TABLE_FEATURES) {
				replies.add((OFTableFeaturesStatsReply) m);
				if (!((OFTableFeaturesStatsReply)m).getFlags().contains(OFStatsReplyFlags.REPLY_MORE)) {
					handleTableFeaturesMessage(replies, false);
					nextState();
				} 
			} else {
				log.error("Received {} message but expected TABLE_FEATURES.", m.getStatsType().toString());
			}
		}
		@Override
		void processOFError(OFErrorMsg m) {
			if ((m.getErrType() == OFErrorType.BAD_REQUEST) &&
					((((OFBadRequestErrorMsg)m).getCode() == OFBadRequestCode.MULTIPART_BUFFER_OVERFLOW)
							|| ((OFBadRequestErrorMsg)m).getCode() == OFBadRequestCode.BAD_STAT)) { 
				log.warn("Switch {} is {} but does not support OFTableFeaturesStats. Assuming all tables can perform any match, action, and instruction in the spec.", 
						sw.getId().toString(), sw.getOFFactory().getVersion().toString());
			} else {
				log.error("Received unexpected OFErrorMsg {} on switch {}.", m.toString(), sw.getId().toString());
			}
			nextState();
		}
		private void nextState() {
			sw.startDriverHandshake();
			if (sw.isDriverHandshakeComplete()) {
				setState(new WaitAppHandshakeState());
			} else {
				setState(new WaitSwitchDriverSubHandshakeState());
			}
		}
		@Override
		void enterState() {
			if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_13) < 0) {
				nextState();
			} else {
				sendHandshakeTableFeaturesRequest();
			}
		}
	}
	public class WaitSwitchDriverSubHandshakeState extends OFSwitchHandshakeState {
		WaitSwitchDriverSubHandshakeState() {
			super(false);
		}
		@Override
		void processOFMessage(OFMessage m) {
			sw.processDriverHandshakeMessage(m);
			if (sw.isDriverHandshakeComplete()) {
				setState(new WaitAppHandshakeState());
			}
		}
		@Override
		void processOFPortStatus(OFPortStatus m) {
			handlePortStatusMessage(m, false);
		}
	}
	public class WaitAppHandshakeState extends OFSwitchHandshakeState {
		private final Iterator<IAppHandshakePluginFactory> pluginIterator;
		private OFSwitchAppHandshakePlugin plugin;
		WaitAppHandshakeState() {
			super(false);
			this.pluginIterator = switchManager.getHandshakePlugins().iterator();
		}
		@Override
		void processOFMessage(OFMessage m) {
			if(m.getType() == OFType.PORT_STATUS){
				OFPortStatus status = (OFPortStatus) m;
				handlePortStatusMessage(status, false);
			}
			else if(plugin != null){
				this.plugin.processOFMessage(m);
			}
			else{
				super.processOFMessage(m);
			}
		}
		void exitPlugin(PluginResult result) {
			if (result.getResultType() == PluginResultType.CONTINUE) {
				if (log.isDebugEnabled()) {
					log.debug("Switch " + getSwitchInfoString() + " app handshake plugin {} returned {}."
							+ " Proceeding normally..",
							this.plugin.getClass().getSimpleName(), result);
				}
				enterNextPlugin();
			} else if (result.getResultType() == PluginResultType.DISCONNECT) {
				log.error("Switch " + getSwitchInfoString() + " app handshake plugin {} returned {}. "
						+ "Disconnecting switch.",
						this.plugin.getClass().getSimpleName(), result);
				mainConnection.disconnect();
			} else if (result.getResultType() == PluginResultType.QUARANTINE) {
				log.warn("Switch " + getSwitchInfoString() + " app handshake plugin {} returned {}. "
						+ "Putting switch into quarantine state.",
						this.plugin.getClass().getSimpleName(),
						result);
				setState(new QuarantineState(result.getReason()));
			}
		}
		@Override
		public void enterState() {
			enterNextPlugin();
		}
		public void enterNextPlugin() {
			if(this.pluginIterator.hasNext()){
				this.plugin = pluginIterator.next().createPlugin();
				this.plugin.init(this, sw, timer);
				this.plugin.enterPlugin();
			}
			else{
				setState(new WaitInitialRoleState());
			}
		}
		@Override
		void processOFPortStatus(OFPortStatus m) {
			handlePortStatusMessage(m, false);
		}
		OFSwitchAppHandshakePlugin getCurrentPlugin() {
			return plugin;
		}
	}
	public class QuarantineState extends OFSwitchHandshakeState {
		private final String quarantineReason;
		QuarantineState(String reason) {
			super(true);
			this.quarantineReason = reason;
		}
		@Override
		public void enterState() {
			setSwitchStatus(SwitchStatus.QUARANTINED);
		}
		@Override
		void processOFPortStatus(OFPortStatus m) {
			handlePortStatusMessage(m, false);
		}
		public String getQuarantineReason() {
			return this.quarantineReason;
		}
	}
	public class WaitInitialRoleState extends OFSwitchHandshakeState {
		WaitInitialRoleState() {
			super(false);
		}
		@Override
		void processOFError(OFErrorMsg m) {
			boolean didHandle = roleChanger.deliverError(m);
			if (!didHandle) {
				logError(m);
			}
		}
		@Override
		void processOFExperimenter(OFExperimenter m) {
			OFControllerRole role = extractNiciraRoleReply(m);
			if (role != null) {
				roleChanger.deliverRoleReply(m.getXid(), role);
			} else {
				unhandledMessageReceived(m);
			}
		}
		@Override
		void processOFRoleReply(OFRoleReply m) {
			roleChanger.deliverRoleReply(m.getXid(), m.getRole());
		}
		@Override
		void processOFStatsReply(OFStatsReply m) {
			illegalMessageReceived(m);
		}
		@Override
		void processOFPortStatus(OFPortStatus m) {
			handlePortStatusMessage(m, false);
		}
		@Override
		void enterState(){
			sendRoleRequest(roleManager.getOFControllerRole());
		}
	}
	public class MasterState extends OFSwitchHandshakeState {
		MasterState() {
			super(true);
		}
		private long sendBarrier() {
			long xid = handshakeTransactionIds--;
			OFBarrierRequest barrier = factory.buildBarrierRequest()
					.setXid(xid)
					.build();
			return xid;
		}
		@Override
		void enterState() {
			if (OFSwitchManager.clearTablesOnEachTransitionToMaster) {
				log.info("Clearing flow tables of {} on upcoming transition to MASTER.", sw.getId().toString());
				clearAllTables();
				initialRole = OFControllerRole.ROLE_MASTER;
				log.info("Clearing flow tables of {} on upcoming initial role as MASTER.", sw.getId().toString());
				clearAllTables();
			}
			addDefaultFlows();
			sendBarrier();
			setSwitchStatus(SwitchStatus.MASTER);
		}
		@Override
		void processOFError(OFErrorMsg m) {
			boolean didHandle = roleChanger.deliverError(m);
			if (didHandle)
				return;
			if ((m.getErrType() == OFErrorType.BAD_REQUEST) &&
					(((OFBadRequestErrorMsg)m).getCode() == OFBadRequestCode.EPERM)) {
				switchManagerCounters.epermErrorWhileSwitchIsMaster.increment();
				log.warn("Received permission error from switch {} while" +
						"being master. Reasserting master role.",
						getSwitchInfoString());
				reassertRole(OFControllerRole.ROLE_MASTER);
			}
			else if ((m.getErrType() == OFErrorType.FLOW_MOD_FAILED) &&
					(((OFFlowModFailedErrorMsg)m).getCode() == OFFlowModFailedCode.ALL_TABLES_FULL)) {
				sw.setTableFull(true);
			}
			else {
				logError(m);
			}
			dispatchMessage(m);
		}
		@Override
		void processOFExperimenter(OFExperimenter m) {
			OFControllerRole role = extractNiciraRoleReply(m);
			if (role != null) {
				roleChanger.deliverRoleReply(m.getXid(), role);
			} else {
				dispatchMessage(m);
			}
		}
		@Override
		void processOFRoleRequest(OFRoleRequest m) {
			sendRoleRequest(m);
		}
		@Override
		void processOFNiciraControllerRoleRequest(OFNiciraControllerRoleRequest m) {
			OFControllerRole role;
			switch (m.getRole()) {
			case ROLE_MASTER:
				role = OFControllerRole.ROLE_MASTER;
				break;
			case ROLE_SLAVE:
				role = OFControllerRole.ROLE_SLAVE;
				break;
			case ROLE_OTHER:
				role = OFControllerRole.ROLE_EQUAL;
				break;
			default:
				log.error("Attempted to change to invalid Nicira role {}.", m.getRole().toString());
				return;
			}
			sendRoleRequest(OFFactories.getFactory(OFVersion.OF_13).buildRoleRequest()
					.setGenerationId(U64.ZERO)
					.setXid(m.getXid())
					.setRole(role)
					.build());
		}
		@Override
		void processOFRoleReply(OFRoleReply m) {
			roleChanger.deliverRoleReply(m.getXid(), m.getRole());
		}
		@Override
		void processOFPortStatus(OFPortStatus m) {
			handlePortStatusMessage(m, true);
		}
		@Override
		void processOFPacketIn(OFPacketIn m) {
			dispatchMessage(m);
		}
		@Override
		void processOFFlowRemoved(OFFlowRemoved m) {
			dispatchMessage(m);
		}
		@Override
		void processOFStatsReply(OFStatsReply m) {
			super.processOFStatsReply(m);
		}
	}
	public class SlaveState extends OFSwitchHandshakeState {
		SlaveState() {
			super(true);
		}
		@Override
		void enterState() {
			setSwitchStatus(SwitchStatus.SLAVE);
			if (initialRole == null) {
				initialRole = OFControllerRole.ROLE_SLAVE;
			}
		}
		@Override
		void processOFError(OFErrorMsg m) {
			boolean didHandle = roleChanger.deliverError(m);
			if (!didHandle) {
				logError(m);
			}
		}
		@Override
		void processOFStatsReply(OFStatsReply m) {
		}
		@Override
		void processOFPortStatus(OFPortStatus m) {
			handlePortStatusMessage(m, true);
		}
		@Override
		void processOFExperimenter(OFExperimenter m) {
			OFControllerRole role = extractNiciraRoleReply(m);
			if (role != null) {
				roleChanger.deliverRoleReply(m.getXid(), role);
			} else {
				unhandledMessageReceived(m);
			}
		}
		@Override
		void processOFRoleReply(OFRoleReply m) {
			roleChanger.deliverRoleReply(m.getXid(), m.getRole());
		}
		@Override
		void processOFRoleRequest(OFRoleRequest m) {
			sendRoleRequest(m);
		}
		@Override
		void processOFNiciraControllerRoleRequest(OFNiciraControllerRoleRequest m) {
			OFControllerRole role;
			switch (m.getRole()) {
			case ROLE_MASTER:
				role = OFControllerRole.ROLE_MASTER;
				break;
			case ROLE_SLAVE:
				role = OFControllerRole.ROLE_SLAVE;
				break;
			case ROLE_OTHER:
				role = OFControllerRole.ROLE_EQUAL;
				break;
			default:
				log.error("Attempted to change to invalid Nicira role {}.", m.getRole().toString());
				return;
			}
			sendRoleRequest(OFFactories.getFactory(OFVersion.OF_13).buildRoleRequest()
					.setGenerationId(U64.ZERO)
					.setXid(m.getXid())
					.setRole(role)
					.build());
		}
		@Override
		void processOFPacketIn(OFPacketIn m) {
			switchManagerCounters.packetInWhileSwitchIsSlave.increment();
			log.warn("Received PacketIn from switch {} while" +
					"being slave. Reasserting slave role.", sw);
			reassertRole(OFControllerRole.ROLE_SLAVE);
		}
	};
	OFSwitchHandshakeHandler(@Nonnull IOFConnectionBackend connection,
			@Nonnull OFFeaturesReply featuresReply,
			@Nonnull IOFSwitchManager switchManager,
			@Nonnull RoleManager roleManager,
			@Nonnull Timer timer) {
		Preconditions.checkNotNull(connection, "connection");
		Preconditions.checkNotNull(featuresReply, "featuresReply");
		Preconditions.checkNotNull(switchManager, "switchManager");
		Preconditions.checkNotNull(roleManager, "roleManager");
		Preconditions.checkNotNull(timer, "timer");
		Preconditions.checkArgument(connection.getAuxId().equals(OFAuxId.MAIN),
				"connection must be MAIN connection but is %s", connection);
		this.switchManager = switchManager;
		this.roleManager = roleManager;
		this.mainConnection = connection;
		this.auxConnections = new ConcurrentHashMap<OFAuxId, IOFConnectionBackend>();
		this.featuresReply = featuresReply;
		this.timer = timer;
		this.switchManagerCounters = switchManager.getCounters();
		this.factory = OFFactories.getFactory(featuresReply.getVersion());
		this.roleChanger = new RoleChanger(DEFAULT_ROLE_TIMEOUT_NS);
		setState(new InitState());
		this.pendingPortStatusMsg = new ArrayList<OFPortStatus>();
		connection.setListener(this);
	}
	public void beginHandshake() {
		Preconditions.checkState(state instanceof InitState, "must be in InitState");
		if (this.featuresReply.getNTables() > 1) {
			log.debug("Have {} table(s) for switch {}", this.featuresReply.getNTables(),
					getSwitchInfoString());
		}
		if (this.featuresReply.getVersion().compareTo(OFVersion.OF_13) < 0) {
			setState(new WaitConfigReplyState());
		} else {
			setState(new WaitPortDescStatsReplyState());
		}
	}
	public DatapathId getDpid(){
		return this.featuresReply.getDatapathId();
	}
	public OFAuxId getOFAuxId(){
		return this.featuresReply.getAuxiliaryId();
	}
	public boolean isHandshakeComplete() {
		return this.state.isHandshakeComplete();
	}
	void sendRoleRequestIfNotPending(OFControllerRole role) {
		try {
			roleChanger.sendRoleRequestIfNotPending(role, 0);
		} catch (IOException e) {
			log.error("Disconnecting switch {} due to IO Error: {}",
					getSwitchInfoString(), e.getMessage());
			mainConnection.disconnect();
		}
	}
	void sendRoleRequestIfNotPending(OFRoleRequest role) {
		try {
			roleChanger.sendRoleRequestIfNotPending(role.getRole(), 0);
		} catch (IOException e) {
			log.error("Disconnecting switch {} due to IO Error: {}",
					getSwitchInfoString(), e.getMessage());
			mainConnection.disconnect();
		}
	}
	void sendRoleRequest(OFControllerRole role) {
		try {
			roleChanger.sendRoleRequest(role, 0);
		} catch (IOException e) {
			log.error("Disconnecting switch {} due to IO Error: {}",
					getSwitchInfoString(), e.getMessage());
			mainConnection.disconnect();
		}
	}
	void sendRoleRequest(OFRoleRequest role) {
		try {
			roleChanger.sendRoleRequest(role.getRole(), role.getXid());
		} catch (IOException e) {
			log.error("Disconnecting switch {} due to IO Error: {}",
					getSwitchInfoString(), e.getMessage());
			mainConnection.disconnect();
		}
	}
	private void dispatchMessage(OFMessage m) {
		this.switchManager.handleMessage(this.sw, m, null);
	}
	private String getSwitchInfoString() {
		if (sw != null)
			return sw.toString();
		String channelString;
		if (mainConnection == null || mainConnection.getRemoteInetAddress() == null) {
			channelString = "?";
		} else {
			channelString = mainConnection.getRemoteInetAddress().toString();
		}
		String dpidString;
		if (featuresReply == null) {
			dpidString = "?";
		} else {
			dpidString = featuresReply.getDatapathId().toString();
		}
		return String.format("[%s DPID[%s]]", channelString, dpidString);
	}
	private void setState(OFSwitchHandshakeState state) {
		this.state = state;
		state.logState();
		state.enterState();
	}
	public void processOFMessage(OFMessage m) {
		state.processOFMessage(m);
	}
	public void processWrittenOFMessage(OFMessage m) {
		state.processWrittenOFMessage(m);
	}
	private void sendHandshakeSetConfig() {
		OFSetConfig configSet = factory.buildSetConfig()
				.setXid(handshakeTransactionIds--)
				.setMissSendLen(0xffff)
				.build();
		OFBarrierRequest barrier = factory.buildBarrierRequest()
				.setXid(handshakeTransactionIds--)
				.build();
		OFGetConfigRequest configReq = factory.buildGetConfigRequest()
				.setXid(handshakeTransactionIds--)
				.build();
		List<OFMessage> msgList = ImmutableList.<OFMessage>of(configSet, barrier, configReq);
		mainConnection.write(msgList);
	}
	protected void sendPortDescRequest() {
		mainConnection.write(factory.portDescStatsRequest(ImmutableSet.<OFStatsRequestFlags>of()));
	}
	private void sendHandshakeDescriptionStatsRequest() {
		OFDescStatsRequest descStatsRequest = factory.buildDescStatsRequest()
				.setXid(handshakeTransactionIds--)
				.build();
		mainConnection.write(descStatsRequest);
	}
	private void sendHandshakeTableFeaturesRequest() {
		OFTableFeaturesStatsRequest tfsr = factory.buildTableFeaturesStatsRequest()
				.setXid(handshakeTransactionIds--)
				.build();
		mainConnection.write(tfsr);
	}
	OFSwitchHandshakeState getStateForTesting() {
		return state;
	}
	void reassertRole(OFControllerRole role){
		this.roleManager.reassertRole(this, HARole.ofOFRole(role));
	}
	void useRoleChangerWithOtherTimeoutForTesting(long roleTimeoutMs) {
		roleChanger = new RoleChanger(TimeUnit.MILLISECONDS.toNanos(roleTimeoutMs));
	}
	public synchronized void auxConnectionOpened(IOFConnectionBackend connection) {
		if(log.isDebugEnabled())
			log.debug("[{}] - Switch Handshake - new aux connection {}", this.getDpid(), connection.getAuxId());
		if (this.getState().equals("ACTIVE") || this.getState().equals("STANDBY")) {
			auxConnections.put(connection.getAuxId(), connection);
			connection.setListener(OFSwitchHandshakeHandler.this);
			log.info("Auxiliary connection {} added for {}.", connection.getAuxId().getValue(), connection.getDatapathId().toString());
		} else {
			log.info("Auxiliary connection {} initiated for {} before main connection handshake complete. Ignorning aux connection attempt.", connection.getAuxId().getValue(), connection.getDatapathId().toString());
		}
	}
	public IOFConnectionBackend getMainConnection() {
		return this.mainConnection;
	}
	public boolean hasConnection(IOFConnectionBackend connection) {
		if (this.mainConnection.equals(connection)
				|| this.auxConnections.get(connection.getAuxId()) == connection) {
			return true;
		} else {
			return false;
		}
	}
	void cleanup() {
		for (IOFConnectionBackend conn : this.auxConnections.values()) {
			conn.disconnect();
		}
		this.mainConnection.disconnect();
	}
	public String getState() {
		return this.state.getClass().getSimpleName();
	}
	public String getQuarantineReason() {
		if(this.state instanceof QuarantineState) {
			QuarantineState qs = (QuarantineState) this.state;
			return qs.getQuarantineReason();
		}
		return null;
	}
	public ImmutableList<IOFConnection> getConnections() {
		ImmutableList.Builder<IOFConnection> builder = ImmutableList.builder();
		builder.add(mainConnection);
		builder.addAll(auxConnections.values());
		return builder.build();
	}
	@Override
	public void connectionClosed(IOFConnectionBackend connection) {
		cleanup();
		if (connection == this.mainConnection) {
			switchManager.handshakeDisconnected(connection.getDatapathId());
			if(sw != null) {
				log.debug("[{}] - main connection {} closed - disconnecting switch",
						connection);
				setSwitchStatus(SwitchStatus.DISCONNECTED);
				switchManager.switchDisconnected(sw);
			}
		}
	}
	@Override
	public void messageReceived(IOFConnectionBackend connection, OFMessage m) {
		processOFMessage(m);
	}
	@Override
	public void messageWritten(IOFConnectionBackend connection, OFMessage m) {
		processWrittenOFMessage(m);
	}
	@Override
	public boolean isSwitchHandshakeComplete(IOFConnectionBackend connection) {
		return state.isHandshakeComplete();
	}
	public void setSwitchStatus(SwitchStatus status) {
		if(sw != null) {
			SwitchStatus oldStatus = sw.getStatus();
			if(oldStatus != status) {
				log.debug("[{}] SwitchStatus change to {} requested, switch is in status " + oldStatus,
						mainConnection.getDatapathId(), status);
				sw.setStatus(status);
				switchManager.switchStatusChanged(sw, oldStatus, status);
			} else {
				log.warn("[{}] SwitchStatus change to {} requested, switch is already in status",
						mainConnection.getDatapathId(), status);
			}
		} else {
			log.warn("[{}] SwitchStatus change to {} requested, but switch is not allocated yet",
					mainConnection.getDatapathId(), status);
		}
	}
}
