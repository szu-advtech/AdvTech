package org.sdnplatform.sync.internal;
import io.netty.util.HashedWheelTimer;
import io.netty.util.Timer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import org.sdnplatform.sync.IClosableIterator;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.IVersion.Occurred;
import org.sdnplatform.sync.error.PersistException;
import org.sdnplatform.sync.error.SyncException;
import org.sdnplatform.sync.error.SyncRuntimeException;
import org.sdnplatform.sync.error.UnknownStoreException;
import org.sdnplatform.sync.internal.StoreRegistry.Hint;
import org.sdnplatform.sync.internal.config.ClusterConfig;
import org.sdnplatform.sync.internal.config.DelegatingCCProvider;
import org.sdnplatform.sync.internal.config.FallbackCCProvider;
import org.sdnplatform.sync.internal.config.IClusterConfigProvider;
import org.sdnplatform.sync.internal.config.Node;
import org.sdnplatform.sync.internal.config.PropertyCCProvider;
import org.sdnplatform.sync.internal.config.StorageCCProvider;
import org.sdnplatform.sync.internal.config.SyncStoreCCProvider;
import org.sdnplatform.sync.internal.rpc.RPCService;
import org.sdnplatform.sync.internal.rpc.TProtocolUtil;
import org.sdnplatform.sync.internal.store.IStorageEngine;
import org.sdnplatform.sync.internal.store.IStore;
import org.sdnplatform.sync.internal.store.MappingStoreListener;
import org.sdnplatform.sync.internal.store.SynchronizingStorageEngine;
import org.sdnplatform.sync.internal.util.ByteArray;
import org.sdnplatform.sync.internal.version.VectorClock;
import org.sdnplatform.sync.thrift.SyncMessage;
import org.sdnplatform.sync.thrift.KeyedValues;
import org.sdnplatform.sync.thrift.KeyedVersions;
import org.sdnplatform.sync.thrift.SyncOfferMessage;
import org.sdnplatform.sync.thrift.SyncValueMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.debugcounter.IDebugCounter;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.threadpool.IThreadPoolService;
public class SyncManager extends AbstractSyncManager {
	protected static final Logger logger =
			LoggerFactory.getLogger(SyncManager.class.getName());
	protected IThreadPoolService threadPool;
	protected IDebugCounterService debugCounter;
	private StoreRegistry storeRegistry = null;
	private Timer timer;
	private IClusterConfigProvider clusterConfigProvider;
	private ClusterConfig clusterConfig = new ClusterConfig();
	protected RPCService rpcService = null;
	private static final int CONFIG_RESCAN_INTERVAL = 10;
	private SingletonTask cleanupTask;
	private SingletonTask antientropyTask;
	private SingletonTask updateConfigTask;
	private static final int SYNC_WORKER_POOL = 2;
	private ExecutorService hintThreadPool;
	private final Random random = new Random();
	private final Map<Integer, Cursor> cursorMap =
			new ConcurrentHashMap<Integer, Cursor>();
	private boolean persistenceEnabled = true;
	private static final String PACKAGE =
			ISyncService.class.getPackage().getName();
	public static IDebugCounter counterHints;
	public static IDebugCounter counterSentValues;
	public static IDebugCounter counterReceivedValues;
	public static IDebugCounter counterPuts;
	public static IDebugCounter counterGets;
	public static IDebugCounter counterIterators;
	public static IDebugCounter counterErrorRemote;
	public static IDebugCounter counterErrorProcessing;
	@Override
	public void registerStore(String storeName, Scope scope) {
		try {
			storeRegistry.register(storeName, scope, false);
		} catch (PersistException e) {
			throw new SyncRuntimeException(e);
		}
	}
	@Override
	public void registerPersistentStore(String storeName, Scope scope)
			throws PersistException {
		storeRegistry.register(storeName, scope, persistenceEnabled);
	}
	public ClusterConfig getClusterConfig() {
		return clusterConfig;
	}
	public void cleanup() throws SyncException {
		for (SynchronizingStorageEngine store : storeRegistry.values()) {
			store.cleanupTask();
		}
	}
	public void antientropy(Node node) {
		if (!rpcService.isConnected(node.getNodeId())) return;
		logger.info("[{}->{}] Synchronizing local state to remote node",
				getLocalNodeId(), node.getNodeId());
		for (SynchronizingStorageEngine store : storeRegistry.values()) {
			if (Scope.LOCAL.equals(store.getScope())) {
				if (node.getDomainId() !=
						getClusterConfig().getNode().getDomainId())
					continue;
			} else if (Scope.UNSYNCHRONIZED.equals(store.getScope())) {
				continue;
			}
			IClosableIterator<Entry<ByteArray,
			List<Versioned<byte[]>>>> entries =
			store.entries();
			try {
				SyncMessage bsm =
						TProtocolUtil.getTSyncOfferMessage(store.getName(),
								store.getScope(),
								store.isPersistent());
				int count = 0;
				while (entries.hasNext()) {
					if (!rpcService.isConnected(node.getNodeId())) return;
					Entry<ByteArray, List<Versioned<byte[]>>> pair =
							entries.next();
					KeyedVersions kv =
							TProtocolUtil.getTKeyedVersions(pair.getKey(),
									pair.getValue());
					bsm.getSyncOffer().addToVersions(kv);
					count += 1;
					if (count >= 50) {
						sendSyncOffer(node.getNodeId(), bsm);
						bsm = TProtocolUtil.getTSyncOfferMessage(store.getName(),
								store.getScope(),
								store.isPersistent());
						count = 0;
					}
				}
				sendSyncOffer(node.getNodeId(), bsm);
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			} finally {
				entries.close();
			}
		}
	}
	public void antientropy() {
		ArrayList<Node> candidates = new ArrayList<Node>();
		for (Node n : clusterConfig.getNodes())
			if (rpcService.isConnected(n.getNodeId()))
				candidates.add(n);
		int numNodes = candidates.size();
		if (numNodes == 0) return;
		Node[] nodes = candidates.toArray(new Node[numNodes]);
		int rn = random.nextInt(numNodes);
		antientropy(nodes[rn]);
	}
	public void writeSyncValue(String storeName, Scope scope,
			boolean persist,
			byte[] key, Iterable<Versioned<byte[]>> values)
					throws PersistException {
		SynchronizingStorageEngine store = storeRegistry.get(storeName);
		if (store == null) {
			store = storeRegistry.register(storeName, scope, persist);
		}
		store.writeSyncValue(new ByteArray(key), values);
	}
	public boolean handleSyncOffer(String storeName,
			byte[] key,
			Iterable<VectorClock> versions)
					throws SyncException {
		SynchronizingStorageEngine store = storeRegistry.get(storeName);
		if (store == null) return true;
		List<Versioned<byte[]>> values = store.get(new ByteArray(key));
		if (values == null || values.size() == 0) return true;
		for (VectorClock vc : versions) {
			for (Versioned<byte[]> value : values) {
				VectorClock existingVc = (VectorClock)value.getVersion();
				if (!vc.compare(existingVc).equals(Occurred.BEFORE))
					return true;
			}
		}
		return false;
	}
	public IStorageEngine<ByteArray, byte[]> getRawStore(String storeName)
			throws UnknownStoreException {
		return getStoreInternal(storeName);
	}
	public IThreadPoolService getThreadPool() {
		return threadPool;
	}
	public void queueSyncTask(SynchronizingStorageEngine e,
			ByteArray key, Versioned<byte[]> value) {
		storeRegistry.queueHint(e.getName(), key, value);
	}
	@Override
	public void addListener(String storeName, MappingStoreListener listener)
			throws UnknownStoreException {
		SynchronizingStorageEngine store = getStoreInternal(storeName);
		store.addListener(listener);
	}
	public void updateConfiguration() {
		if (updateConfigTask != null)
			updateConfigTask.reschedule(500, TimeUnit.MILLISECONDS);
	}
	public Cursor getCursor(int cursorId) {
		return cursorMap.get(Integer.valueOf(cursorId));
	}
	public Cursor newCursor(String storeName) throws UnknownStoreException {
		IStore<ByteArray, byte[]> store = getStore(storeName);
		int cursorId = rpcService.getTransactionId();
		Cursor cursor = new Cursor(cursorId, store.entries());
		cursorMap.put(Integer.valueOf(cursorId), cursor);
		return cursor;
	}
	public void closeCursor(Cursor cursor) {
		cursor.close();
		cursorMap.remove(Integer.valueOf(cursor.getCursorId()));
	}
	@Override
	public IStore<ByteArray,byte[]> getStore(String storeName)
			throws UnknownStoreException {
		return getRawStore(storeName);
	}
	@Override
	public short getLocalNodeId() {
		Node l = clusterConfig.getNode();
		if (l == null) return Short.MAX_VALUE;
		return l.getNodeId();
	}
	@Override
	public void shutdown() {
		logger.info("Shutting down Sync Manager: {} {}",
				clusterConfig.getNode().getHostname(),
				clusterConfig.getNode().getPort());
		if (rpcService != null) {
			rpcService.shutdown();
		}
		if (hintThreadPool != null) {
			hintThreadPool.shutdown();
		}
		if (storeRegistry != null) {
			storeRegistry.shutdown();
		}
		if (timer != null)
            timer.stop();
        timer = null;
		hintThreadPool = null;
		rpcService = null;
	}
	@Override
	public void init(FloodlightModuleContext context)
			throws FloodlightModuleException {
		threadPool = context.getServiceImpl(IThreadPoolService.class);
		debugCounter = context.getServiceImpl(IDebugCounterService.class);
		Map<String, String> config = context.getConfigParams(this);
		storeRegistry = new StoreRegistry(this, config.get("dbPath"));
		String[] configProviders =
			{PropertyCCProvider.class.getName(),
				SyncStoreCCProvider.class.getName(),
				StorageCCProvider.class.getName(),
				FallbackCCProvider.class.getName()};
		try {
			if (config.containsKey("persistenceEnabled")) {
				persistenceEnabled =
						Boolean.parseBoolean(config.get("persistenceEnabled"));
			}
			if (config.containsKey("configProviders")) {
				configProviders = config.get("configProviders").split(",");
			}
			DelegatingCCProvider dprovider = new DelegatingCCProvider();
			for (String configProvider : configProviders) {
				Class<?> cClass = Class.forName(configProvider);
				IClusterConfigProvider provider =
						(IClusterConfigProvider) cClass.newInstance();
				dprovider.addProvider(provider);
			}
			dprovider.init(this, context);
			clusterConfigProvider = dprovider;
		} catch (Exception e) {
			throw new FloodlightModuleException("Could not instantiate config" +
					"providers " + Arrays.toString(configProviders), e);
		}
		String manualStoreString = config.get("manualStores");
		if (manualStoreString != null) {
			List<String> manualStores = null;
			try {
				manualStores =
						(new ObjectMapper()).readValue(manualStoreString,
								new TypeReference<List<String>>() {});
			} catch (Exception e) {
				throw new FloodlightModuleException("Failed to parse sync " +
						"manager manual stores: " + manualStoreString, e);
			}
			for (String s : manualStores) {
				registerStore(s, Scope.GLOBAL);
			}
		}
		registerDebugCounters(context);
	}
	private void registerDebugCounters(FloodlightModuleContext context)
			throws FloodlightModuleException {
		if (context != null) {
			debugCounter.registerModule(PACKAGE);
			counterHints = debugCounter.registerCounter(PACKAGE, "hints",
					"Queued sync events processed");
			counterSentValues = debugCounter.registerCounter(PACKAGE, "sent-values",
					"Values synced to remote node");
			counterReceivedValues = debugCounter.registerCounter(PACKAGE, "received-values",
					"Values received from remote node");
			counterPuts = debugCounter.registerCounter(PACKAGE, "puts",
					"Local puts to store");
			counterGets = debugCounter.registerCounter(PACKAGE, "gets",
					"Local gets from store");
			counterIterators = debugCounter.registerCounter(PACKAGE, "iterators",
					"Local iterators created over store");
			counterErrorRemote = debugCounter.registerCounter(PACKAGE, "error-remote",
					"Number of errors sent from remote clients",
					IDebugCounterService.MetaData.ERROR);
			counterErrorProcessing = debugCounter.registerCounter(PACKAGE,
					"error-processing",
					"Number of errors processing messages from remote clients",
					IDebugCounterService.MetaData.ERROR);
		}
	}
	@Override
	public void startUp(FloodlightModuleContext context)
			throws FloodlightModuleException {
		timer = new HashedWheelTimer();
		rpcService = new RPCService(this, debugCounter, timer);
		cleanupTask = new SingletonTask(threadPool.getScheduledExecutor(),
				new CleanupTask());
		cleanupTask.reschedule(CLEANUP_INTERVAL +
				random.nextInt(30), TimeUnit.SECONDS);
		antientropyTask = new SingletonTask(threadPool.getScheduledExecutor(),
				new AntientropyTask());
		antientropyTask.reschedule(ANTIENTROPY_INTERVAL +
				random.nextInt(30), TimeUnit.SECONDS);
		final ThreadGroup tg = new ThreadGroup("Hint Workers");
		tg.setMaxPriority(Thread.NORM_PRIORITY - 2);
		ThreadFactory f = new ThreadFactory() {
			AtomicInteger id = new AtomicInteger();
			@Override
			public Thread newThread(Runnable runnable) {
				return new Thread(tg, runnable,
						"HintWorker-" + id.getAndIncrement());
			}
		};
		hintThreadPool = Executors.newCachedThreadPool(f);
		for (int i = 0; i < SYNC_WORKER_POOL; i++) {
			hintThreadPool.execute(new HintWorker());
		}
		doUpdateConfiguration();
		rpcService.run();
		updateConfigTask =
				new SingletonTask(threadPool.getScheduledExecutor(),
						new UpdateConfigTask());
		updateConfigTask.reschedule(CONFIG_RESCAN_INTERVAL, TimeUnit.SECONDS);
	}
	@Override
	public Collection<Class<? extends IFloodlightService>>
	getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IThreadPoolService.class);
		l.add(IStorageSourceService.class);
		l.add(IDebugCounterService.class);
		return l;
	}
	protected void doUpdateConfiguration()
			throws FloodlightModuleException {
		try {
			ClusterConfig oldConfig = clusterConfig;
			clusterConfig = clusterConfigProvider.getConfig();
			if (clusterConfig.equals(oldConfig)) return;
			logger.info("[{}] Updating sync configuration {}",
					clusterConfig.getNode().getNodeId(),
					clusterConfig);
			if (oldConfig.getNode() != null &&
					!clusterConfig.getNode().equals(oldConfig.getNode())) {
				logger.info("[{}] Local node configuration changed; restarting sync" +
						"service", oldConfig.getNode().getNodeId());
				shutdown();
				startUp(null);
			}
			for (Node n : clusterConfig.getNodes()) {
				Node existing = oldConfig.getNode(n.getNodeId());
				if (existing != null && !n.equals(existing)) {
					logger.info("[{}->{}] Configuration for node has changed",
							getLocalNodeId(), n.getNodeId());
					rpcService.disconnectNode(n.getNodeId());
				}
			}
			for (Node n : oldConfig.getNodes()) {
				Node nn = clusterConfig.getNode(n.getNodeId());
				if (nn == null) {
					logger.info("[{}->{}] Disconnecting deconfigured node",
							getLocalNodeId(), n.getNodeId());
					rpcService.disconnectNode(n.getNodeId());
				}
			}
		} catch (Exception e) {
			throw new FloodlightModuleException("Could not update " +
					"configuration", e);
		}
	}
	protected SynchronizingStorageEngine getStoreInternal(String storeName)
			throws UnknownStoreException {
		SynchronizingStorageEngine store = storeRegistry.get(storeName);
		if (store == null) {
			throw new UnknownStoreException("Store " + storeName +
					" has not been registered");
		}
		return store;
	}
	private void sendSyncOffer(short nodeId, SyncMessage bsm)
			throws InterruptedException {
		SyncOfferMessage som = bsm.getSyncOffer();
		if (!som.isSetVersions()) return;
		if (logger.isTraceEnabled()) {
			logger.trace("[{}->{}] Sending SyncOffer with {} elements",
					new Object[]{getLocalNodeId(), nodeId,
					som.getVersionsSize()});
		}
		som.getHeader().setTransactionId(rpcService.getTransactionId());
		rpcService.writeToNode(nodeId, bsm);
	}
	protected class CleanupTask implements Runnable {
		@Override
		public void run() {
			try {
				if (rpcService != null)
					cleanup();
			} catch (Exception e) {
				logger.error("Cleanup task failed", e);
			}
			if (rpcService != null) {
				cleanupTask.reschedule(CLEANUP_INTERVAL +
						random.nextInt(30), TimeUnit.SECONDS);
			}
		}
	}
	protected class AntientropyTask implements Runnable {
		@Override
		public void run() {
			try {
				if (rpcService != null)
					antientropy();
			} catch (Exception e) {
				logger.error("Antientropy task failed", e);
			}
			if (rpcService != null) {
				antientropyTask.reschedule(ANTIENTROPY_INTERVAL +
						random.nextInt(30),
						TimeUnit.SECONDS);
			}
		}
	}
	protected class UpdateConfigTask implements Runnable {
		@Override
		public void run() {
			try {
				if (rpcService != null)
					doUpdateConfiguration();
			} catch (Exception e) {
				logger.error("Failed to update configuration", e);
			}
			if (rpcService != null) {
				updateConfigTask.reschedule(CONFIG_RESCAN_INTERVAL,
						TimeUnit.SECONDS);
			}
		}
	}
	protected class HintWorker implements Runnable {
		ArrayList<Hint> tasks = new ArrayList<Hint>(50);
		protected Map<String, SyncMessage> messages =
				new LinkedHashMap<String, SyncMessage>();
		@Override
		public void run() {
			while (rpcService != null) {
				try {
					storeRegistry.takeHints(tasks, 50);
					for (Hint task : tasks) {
						counterHints.increment();
						SynchronizingStorageEngine store =
								storeRegistry.get(task.getHintKey().
										getStoreName());
						SyncMessage bsm = getMessage(store);
						KeyedValues kv =
								TProtocolUtil.
								getTKeyedValues(task.getHintKey().getKey(),
										task.getValues());
						bsm.getSyncValue().addToValues(kv);
					}
					Iterable<Node> nodes = getClusterConfig().getNodes();
					short localDomainId =
							getClusterConfig().getNode().getDomainId();
					short localNodeId =
							getClusterConfig().getNode().getNodeId();
					for (Node n : nodes) {
						if (localNodeId == n.getNodeId())
							continue;
						for (SyncMessage bsm : messages.values()) {
							SyncValueMessage svm = bsm.getSyncValue();
							if (svm.getStore().getScope().
									equals(org.sdnplatform.sync.thrift.
											Scope.LOCAL) &&
											n.getDomainId() != localDomainId) {
								continue;
							}
							svm.getHeader().setTransactionId(rpcService.getTransactionId());
							counterSentValues.add(bsm.getSyncValue().getValuesSize());
							rpcService.writeToNode(n.getNodeId(), bsm);
						}
					}
					tasks.clear();
					clearMessages();
				} catch (Exception e) {
					logger.error("Error occured in synchronization worker", e);
				}
			}
		}
		private void clearMessages() {
			messages.clear();
		}
		private SyncMessage getMessage(SynchronizingStorageEngine store) {
			String storeName = store.getName();
			SyncMessage bsm = messages.get(storeName);
			if (bsm == null) {
				bsm = TProtocolUtil.getTSyncValueMessage(storeName,
						store.getScope(),
						store.isPersistent());
				messages.put(storeName, bsm);
			}
			return bsm;
		}
	}
}
