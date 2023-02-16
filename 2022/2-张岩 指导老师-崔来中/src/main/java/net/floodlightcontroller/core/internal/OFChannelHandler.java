package net.floodlightcontroller.core.internal;
import java.io.IOException;
import java.nio.channels.ClosedChannelException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.RejectedExecutionException;
import javax.annotation.Nonnull;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.timeout.IdleStateHandler;
import io.netty.handler.timeout.ReadTimeoutException;
import io.netty.util.AttributeKey;
import io.netty.util.Timer;
import net.floodlightcontroller.core.IOFConnectionBackend;
import net.floodlightcontroller.core.internal.OFChannelInitializer.PipelineHandler;
import net.floodlightcontroller.core.internal.OFChannelInitializer.PipelineHandshakeTimeout;
import net.floodlightcontroller.core.internal.OFChannelInitializer.PipelineIdleReadTimeout;
import net.floodlightcontroller.core.internal.OFChannelInitializer.PipelineIdleWriteTimeout;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import org.projectfloodlight.openflow.exceptions.OFParseError;
import org.projectfloodlight.openflow.protocol.OFEchoReply;
import org.projectfloodlight.openflow.protocol.OFEchoRequest;
import org.projectfloodlight.openflow.protocol.OFErrorMsg;
import org.projectfloodlight.openflow.protocol.OFExperimenter;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
import org.projectfloodlight.openflow.protocol.OFFeaturesRequest;
import org.projectfloodlight.openflow.protocol.OFHello;
import org.projectfloodlight.openflow.protocol.OFHelloElem;
import org.projectfloodlight.openflow.protocol.OFHelloElemVersionbitmap;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPortStatus;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.ver13.OFHelloElemTypeSerializerVer13;
import org.projectfloodlight.openflow.protocol.ver14.OFHelloElemTypeSerializerVer14;
import org.projectfloodlight.openflow.types.OFAuxId;
import org.projectfloodlight.openflow.types.U32;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.base.Preconditions;
class OFChannelHandler extends SimpleChannelInboundHandler<Iterable<OFMessage>> {
	private static final Logger log = LoggerFactory.getLogger(OFChannelHandler.class);
	public static final AttributeKey<OFChannelInfo> ATTR_CHANNEL_INFO = AttributeKey.valueOf("channelInfo");
	private final ChannelPipeline pipeline;
	private final INewOFConnectionListener newConnectionListener;
	private final SwitchManagerCounters counters;
	private Channel channel;
	private final Timer timer;
	private volatile OFChannelState state;
	private OFFactory factory;
	private OFFeaturesReply featuresReply;
	private volatile OFConnection connection;
	private final IDebugCounterService debugCounters;
	private final List<U32> ofBitmaps;
	private long handshakeTransactionIds = 0x00FFFFFFFFL;
	private volatile long echoSendTime;
	private volatile long featuresLatency;
	public abstract class OFChannelState {
		void processOFHello(OFHello m)
				throws IOException {
			illegalMessageReceived(m);
		}
		void processOFEchoRequest(OFEchoRequest m)
				throws IOException {
			sendEchoReply(m);
		}
		void processOFEchoReply(OFEchoReply m)
				throws IOException {
			updateLatency(U64.of( (System.currentTimeMillis() - echoSendTime) / 2) );
		}
		void processOFError(OFErrorMsg m) {
			logErrorDisconnect(m); 
		}
		void processOFExperimenter(OFExperimenter m) {
			unhandledMessageReceived(m);
		}
		void processOFFeaturesReply(OFFeaturesReply  m)
				throws IOException {
			illegalMessageReceived(m);
		}
		void processOFPortStatus(OFPortStatus m) {
			unhandledMessageReceived(m);
		}
		private final boolean channelHandshakeComplete;
		OFChannelState(boolean handshakeComplete) {
			this.channelHandshakeComplete = handshakeComplete;
		}
		void logState() {
			log.debug("{} OFConnection Handshake - enter state {}",
					getConnectionInfoString(), this.getClass().getSimpleName());
		}
		void enterState() throws IOException{
		}
		protected String getSwitchStateMessage(OFMessage m,
				String details) {
			return String.format("Switch: [%s], State: [%s], received: [%s]"
					+ ", details: %s",
					getConnectionInfoString(),
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
			counters.unhandledMessage.increment();
			if (log.isDebugEnabled()) {
				String msg = getSwitchStateMessage(m,
						"Ignoring unexpected message");
				log.debug(msg);
			}
		}
		protected void logError(OFErrorMsg error) {
			log.error("{} from switch {} in state {}",
					new Object[] {
					error.toString(),
					getConnectionInfoString(),
					this.toString()});
		}
		protected void logErrorDisconnect(OFErrorMsg error) {
			logError(error);
			channel.disconnect();
		}
		void processOFMessage(OFMessage m)
				throws IOException {
			if (!state.channelHandshakeComplete) {
				switch(m.getType()) {
				case HELLO:
					processOFHello((OFHello)m);
					break;
				case ERROR:
					processOFError((OFErrorMsg)m);
					break;
				case FEATURES_REPLY:
					processOFFeaturesReply((OFFeaturesReply)m);
					break;
				case EXPERIMENTER:
					processOFExperimenter((OFExperimenter)m);
					break;
				case ECHO_REPLY:
					processOFEchoReply((OFEchoReply)m);
					break;
				case ECHO_REQUEST:
					processOFEchoRequest((OFEchoRequest)m);
					break;
				case PORT_STATUS:
					processOFPortStatus((OFPortStatus)m);
					break;
				default:
					illegalMessageReceived(m);
					break;
				}
			}
			else{
				switch(m.getType()){
				case ECHO_REPLY:
					processOFEchoReply((OFEchoReply)m);
					break;
				case ECHO_REQUEST:
					processOFEchoRequest((OFEchoRequest)m);
					break;
				default:
					sendMessageToConnection(m);
					break;
				}
			}
		}
	}
	class InitState extends OFChannelState {
		InitState() {
			super(false);
		}
	}
	class WaitHelloState extends OFChannelState {
		WaitHelloState() {
			super(false);
		}
		@Override
		void processOFHello(OFHello m) throws IOException {
			OFVersion theirVersion = m.getVersion();
			OFVersion commonVersion = null;
			if (theirVersion.compareTo(OFVersion.OF_13) >= 0 && !m.getElements().isEmpty()) {
				List<U32> bitmaps = new ArrayList<U32>();
				List<OFHelloElem> elements = m.getElements();
				for (OFHelloElem e : elements) {
					if (m.getVersion().equals(OFVersion.OF_13) 
							&& e.getType() == OFHelloElemTypeSerializerVer13.VERSIONBITMAP_VAL) {
						bitmaps.addAll(((OFHelloElemVersionbitmap) e).getBitmaps());
					} else if (m.getVersion().equals(OFVersion.OF_14) 
							&& e.getType() == OFHelloElemTypeSerializerVer14.VERSIONBITMAP_VAL) {
						bitmaps.addAll(((OFHelloElemVersionbitmap) e).getBitmaps());
					}
				}
				commonVersion = computeOFVersionFromBitmap(bitmaps);
				if (commonVersion == null) {
					log.error("Could not negotiate common OpenFlow version for {} with greatest version bitmap algorithm.", channel.remoteAddress());
					channel.disconnect();
					return;
				} else {
					log.info("Negotiated OpenFlow version of {} for {} with greatest version bitmap algorithm.", commonVersion.toString(), channel.remoteAddress());
					factory = OFFactories.getFactory(commonVersion);
					OFMessageDecoder decoder = pipeline.get(OFMessageDecoder.class);
					decoder.setVersion(commonVersion);
				}
			}
			else if (theirVersion.compareTo(factory.getVersion()) < 0) {
				log.info("Negotiated down to switch OpenFlow version of {} for {} using lesser hello header algorithm.", theirVersion.toString(), channel.remoteAddress());
				factory = OFFactories.getFactory(theirVersion);
				OFMessageDecoder decoder = pipeline.get(OFMessageDecoder.class);
				decoder.setVersion(theirVersion);
			else if (theirVersion.equals(factory.getVersion())) {
				log.info("Negotiated equal OpenFlow version of {} for {} using lesser hello header algorithm.", factory.getVersion().toString(), channel.remoteAddress());
			}
			else {
				log.info("Negotiated down to controller OpenFlow version of {} for {} using lesser hello header algorithm.", factory.getVersion().toString(), channel.remoteAddress());
			}
			setState(new WaitFeaturesReplyState());
		}
		@Override
		void enterState() throws IOException {
			sendHelloMessage();
		}
	}
	class WaitFeaturesReplyState extends OFChannelState {
		WaitFeaturesReplyState() {
			super(false);
		}
		@Override
		void processOFFeaturesReply(OFFeaturesReply  m)
				throws IOException {
			featuresReply = m;
			featuresLatency = (System.currentTimeMillis() - featuresLatency) / 2;
			setState(new CompleteState());
		}
		@Override
		void processOFHello(OFHello m) throws IOException {
			if (m.getVersion().equals(factory.getVersion())) {
				log.warn("Ignoring second hello from {} in state {}. Might be a Brocade.", channel.remoteAddress(), state.toString());
			} else {
			}
		}
		@Override
		void processOFPortStatus(OFPortStatus m) {
			log.warn("Ignoring PORT_STATUS message from {} during OpenFlow channel establishment. Ports will be explicitly queried in a later state.", channel.remoteAddress());
		}
		@Override
		void enterState() throws IOException {
			sendFeaturesRequest();
			featuresLatency = System.currentTimeMillis();
		}
		@Override
		void processOFMessage(OFMessage m) throws IOException {
			if (m.getType().equals(OFType.PACKET_IN)) {
				log.warn("Ignoring PACKET_IN message from {} during OpenFlow channel establishment.", channel.remoteAddress());
			} else {
				super.processOFMessage(m);
			}
		}
	};
	class CompleteState extends OFChannelState{
		CompleteState() {
			super(true);
		}
		@Override
		void enterState() throws IOException{
			setSwitchHandshakeTimeout();
			if (featuresReply.getVersion().compareTo(OFVersion.OF_13) < 0){
				connection = new OFConnection(featuresReply.getDatapathId(), factory, channel, OFAuxId.MAIN, debugCounters, timer);
			}
			else {
				connection = new OFConnection(featuresReply.getDatapathId(), factory, channel, featuresReply.getAuxiliaryId(), debugCounters, timer);
				if (!featuresReply.getAuxiliaryId().equals(OFAuxId.MAIN)) {
					setAuxChannelIdle();
				}
			}
			connection.updateLatency(U64.of(featuresLatency));
			echoSendTime = 0;
			notifyConnectionOpened(connection);
		}
	};
	OFChannelHandler(@Nonnull IOFSwitchManager switchManager,
			@Nonnull INewOFConnectionListener newConnectionListener,
			@Nonnull ChannelPipeline pipeline,
			@Nonnull IDebugCounterService debugCounters,
			@Nonnull Timer timer,
			@Nonnull List<U32> ofBitmaps,
			@Nonnull OFFactory defaultFactory) {
		Preconditions.checkNotNull(switchManager, "switchManager");
		Preconditions.checkNotNull(newConnectionListener, "connectionOpenedListener");
		Preconditions.checkNotNull(pipeline, "pipeline");
		Preconditions.checkNotNull(timer, "timer");
		Preconditions.checkNotNull(debugCounters, "debugCounters");
		this.pipeline = pipeline;
		this.debugCounters = debugCounters;
		this.newConnectionListener = newConnectionListener;
		this.counters = switchManager.getCounters();
		this.state = new InitState();
		this.timer = timer;
		this.ofBitmaps = ofBitmaps;
		this.factory = defaultFactory;
		log.debug("constructor on OFChannelHandler {}", String.format("%08x", System.identityHashCode(this)));
	}
	private OFVersion computeOFVersionFromBitmap(List<U32> theirs) {		
		Iterator<U32> theirsItr = theirs.iterator();
		Iterator<U32> oursItr = ofBitmaps.iterator();
		OFVersion version = null;
		int pos = 0;
		int size = 32;
		while (theirsItr.hasNext() && oursItr.hasNext()) {
			int t = theirsItr.next().getRaw();
			int o = oursItr.next().getRaw();
							version = v;
						}
					}
				}
			}
		}
		return version;
	}
	public boolean isSwitchHandshakeComplete() {
		if (this.state.channelHandshakeComplete) {
			return connection.getListener().isSwitchHandshakeComplete(connection);
		} else {
			return false;
		}
	}
	private final void notifyConnectionOpened(OFConnection connection){
		this.connection = connection;
		this.newConnectionListener.connectionOpened(connection, featuresReply);
	}
	private final void notifyConnectionClosed(OFConnection connection){
		connection.getListener().connectionClosed(connection);
	}
	private final void sendMessageToConnection(OFMessage m) {
		connection.messageReceived(m);
	}
	@Override
	public void channelActive(ChannelHandlerContext ctx) throws Exception {
		log.debug("channelConnected on OFChannelHandler {}", String.format("%08x", System.identityHashCode(this)));
		counters.switchConnected.increment();
		channel = ctx.channel();
		log.info("New switch connection from {}", channel.remoteAddress());
		setState(new WaitHelloState());
	}
	@Override
	public void channelInactive(ChannelHandlerContext ctx) {
		if (this.connection != null) {
			this.connection.disconnected();
			notifyConnectionClosed(this.connection);
		}
		log.info("[{}] Disconnected connection", getConnectionInfoString());
	}
	@Override
	public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause)
			throws Exception {
		if (cause instanceof ReadTimeoutException) {
			if (featuresReply.getVersion().compareTo(OFVersion.OF_13) < 0) {
				log.error("Disconnecting switch {} due to read timeout on main cxn.",
						getConnectionInfoString());
				ctx.channel().close();
			} else {
				if (featuresReply.getAuxiliaryId().equals(OFAuxId.MAIN)) {
					log.error("Disconnecting switch {} due to read timeout on main cxn.",
							getConnectionInfoString());
					ctx.channel().close();
				} else {
					log.warn("Switch {} encountered read timeout on aux cxn.",
							getConnectionInfoString());
				}
			}
			counters.switchDisconnectReadTimeout.increment();
		} else if (cause instanceof HandshakeTimeoutException) {
			log.error("Disconnecting switch {}: failed to complete handshake. Channel handshake complete : {}",
					getConnectionInfoString(),
					this.state.channelHandshakeComplete);
			counters.switchDisconnectHandshakeTimeout.increment();
			ctx.channel().close();
		} else if (cause instanceof ClosedChannelException) {
			log.debug("Channel for sw {} already closed", getConnectionInfoString());
		} else if (cause instanceof IOException) {
			log.error("Disconnecting switch {} due to IO Error: {}",
					getConnectionInfoString(), cause.getMessage());
			if (log.isDebugEnabled()) {
				log.debug("StackTrace for previous Exception: ", cause);
			}
			counters.switchDisconnectIOError.increment();
			ctx.channel().close();
		} else if (cause instanceof SwitchStateException) {
			log.error("Disconnecting switch {} due to switch state error: {}",
					getConnectionInfoString(), cause.getMessage());
			if (log.isDebugEnabled()) {
				log.debug("StackTrace for previous Exception: ", cause);
			}
			counters.switchDisconnectSwitchStateException.increment();
			ctx.channel().close();
		} else if (cause instanceof OFAuxException) {
			log.error("Disconnecting switch {} due to OF Aux error: {}",
					getConnectionInfoString(), cause.getMessage());
			if (log.isDebugEnabled()) {
				log.debug("StackTrace for previous Exception: ", cause);
			}
			counters.switchDisconnectSwitchStateException.increment();
			ctx.channel().close();
		} else if (cause instanceof OFParseError) {
			log.error("Disconnecting switch "
					+ getConnectionInfoString() +
					" due to message parse failure",
					cause);
			counters.switchDisconnectParseError.increment();
			ctx.channel().close();
		} else if (cause instanceof RejectedExecutionException) {
			log.warn("Could not process message: queue full");
			counters.rejectedExecutionException.increment();
		} else if (cause instanceof IllegalArgumentException) {
			log.error("Illegal argument exception with switch {}. {}", getConnectionInfoString(), cause);
			counters.switchSslConfigurationError.increment();
			ctx.channel().close();
		} else {
			log.error("Error while processing message from switch "
					+ getConnectionInfoString()
					+ "state " + this.state, cause);
			counters.switchDisconnectOtherException.increment();
			ctx.channel().close();
		}
	}
	@Override
	public void userEventTriggered(ChannelHandlerContext ctx, Object evt) throws Exception {
		log.debug("channelIdle on OFChannelHandler {}", String.format("%08x", System.identityHashCode(this)));
		OFChannelHandler handler = ctx.pipeline().get(OFChannelHandler.class);
		handler.sendEchoRequest();
	}
	@Override
	public void channelRead0(ChannelHandlerContext ctx, Iterable<OFMessage> msgList) throws Exception {
		for (OFMessage ofm : msgList) {
			try {
				state.processOFMessage(ofm);
			}
			catch (Exception ex) {
				ctx.fireExceptionCaught(ex);
			}
		}
	}
	private void setAuxChannelIdle() {
		IdleStateHandler idleHandler = new IdleStateHandler(
				PipelineIdleReadTimeout.AUX,
				PipelineIdleWriteTimeout.AUX,
				0);
		pipeline.replace(PipelineHandler.MAIN_IDLE,
				PipelineHandler.AUX_IDLE,
				idleHandler);
	}
	private void setSwitchHandshakeTimeout() {
		HandshakeTimeoutHandler handler = new HandshakeTimeoutHandler(
				this,
				this.timer,
				PipelineHandshakeTimeout.SWITCH);
		pipeline.replace(PipelineHandler.CHANNEL_HANDSHAKE_TIMEOUT,
				PipelineHandler.SWITCH_HANDSHAKE_TIMEOUT, handler);
	}
	private String getConnectionInfoString() {
		String channelString;
		if (channel == null || channel.remoteAddress() == null) {
			channelString = "?";
		} else {
			channelString = channel.remoteAddress().toString();
			if(channelString.startsWith("/"))
				channelString = channelString.substring(1);
		}
		String dpidString;
		if (featuresReply == null) {
			dpidString = "?";
		} else {
			StringBuilder b = new StringBuilder();
			b.append(featuresReply.getDatapathId());
			if(featuresReply.getVersion().compareTo(OFVersion.OF_13) >= 0) {
				b.append("(").append(featuresReply.getAuxiliaryId()).append(")");
			}
			dpidString = b.toString();
		}
		return String.format("[%s from %s]", dpidString, channelString );
	}
	private void setState(OFChannelState state) throws IOException {
		this.state = state;
		state.logState();
		state.enterState();
	}
	private void sendFeaturesRequest() throws IOException {
		OFFeaturesRequest m = factory.buildFeaturesRequest()
				.setXid(handshakeTransactionIds--)
				.build();
		write(m);
	}
	private void sendHelloMessage() throws IOException {
		OFHello.Builder builder = factory.buildHello();
		if (factory.getVersion().compareTo(OFVersion.OF_13) >= 0) {
			List<OFHelloElem> he = new ArrayList<OFHelloElem>();
			he.add(factory.buildHelloElemVersionbitmap()
					.setBitmaps(ofBitmaps)
					.build());
			builder.setElements(he);
		}
		OFHello m = builder.setXid(handshakeTransactionIds--)
				.build();
		write(m);
		log.debug("Send hello: {}", m); 
	}
	private void sendEchoRequest() {
		OFEchoRequest request = factory.buildEchoRequest()
				.setXid(handshakeTransactionIds--)
				.build();
		echoSendTime = System.currentTimeMillis();
		write(request);
	}
	private void sendEchoReply(OFEchoRequest request) {
		OFEchoReply reply = factory.buildEchoReply()
				.setXid(request.getXid())
				.setData(request.getData())
				.build();
		write(reply);
	}
	private void write(OFMessage m) {
		channel.writeAndFlush(Collections.singletonList(m));
	}
	OFChannelState getStateForTesting() {
		return state;
	}
	IOFConnectionBackend getConnectionForTesting() {
		return connection;
	}
	ChannelPipeline getPipelineForTesting() {
		return this.pipeline;
	}
	private void updateLatency(U64 latency) {
		if (connection != null) {
			connection.updateLatency(latency);
		}
	}
}
