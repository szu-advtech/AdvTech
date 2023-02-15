package net.floodlightcontroller.topology;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.test.MockFloodlightProvider;
import net.floodlightcontroller.core.test.MockSwitchManager;
import net.floodlightcontroller.core.test.MockThreadPoolService;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.MockDebugCounterService;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.debugevent.MockDebugEventService;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery;
import net.floodlightcontroller.linkdiscovery.ILinkDiscoveryService;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import net.floodlightcontroller.topology.NodePortTuple;
import net.floodlightcontroller.topology.TopologyInstance;
import net.floodlightcontroller.topology.TopologyManager;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class TopologyInstanceTest {
    protected static Logger log = LoggerFactory.getLogger(TopologyInstanceTest.class);
    protected TopologyManager topologyManager;
    protected FloodlightModuleContext fmc;
    protected ILinkDiscoveryService linkDiscovery;
    protected MockFloodlightProvider mockFloodlightProvider;
    protected int DIRECT_LINK = 1;
    protected int MULTIHOP_LINK = 2;
    protected int TUNNEL_LINK = 3;
    @Before 
    public void SetUp() throws Exception {
        fmc = new FloodlightModuleContext();
        linkDiscovery = EasyMock.createMock(ILinkDiscoveryService.class);
        mockFloodlightProvider = new MockFloodlightProvider();
        fmc.addService(IFloodlightProviderService.class, mockFloodlightProvider);
        fmc.addService(IOFSwitchService.class, new MockSwitchManager());
        fmc.addService(ILinkDiscoveryService.class, linkDiscovery);
        fmc.addService(IDebugCounterService.class, new MockDebugCounterService());
        fmc.addService(IDebugEventService.class, new MockDebugEventService());
        MockThreadPoolService tp = new MockThreadPoolService();
        topologyManager = new TopologyManager();
        fmc.addService(IThreadPoolService.class, tp);
        topologyManager.init(fmc);
        tp.init(fmc);
        tp.startUp(fmc);
    }
    protected void verifyClusters(int[][] clusters) {
        verifyClusters(clusters, true);
    }
    protected void verifyClusters(int[][] clusters, boolean tunnelsEnabled) {
        List<DatapathId> verifiedSwitches = new ArrayList<DatapathId>();
        for (int i = 0; i < clusters.length; i++)
            Arrays.sort(clusters[i]);
        TopologyInstance ti = 
                topologyManager.getCurrentInstance(tunnelsEnabled);
        Set<DatapathId> switches = ti.getSwitches();
        for (DatapathId sw: switches) {
            if (!verifiedSwitches.contains(sw)) {
                int[] expectedCluster = null;
                for (int j = 0; j < clusters.length; j++) {
                    if (Arrays.binarySearch(clusters[j], (int)sw.getLong()) >= 0) {
                        expectedCluster = clusters[j];
                        break;
                    }
                }
                if (expectedCluster != null) {
                    Set<DatapathId> cluster = ti.getSwitchesInOpenflowDomain(sw);
                    assertEquals(expectedCluster.length, cluster.size());
                    for (DatapathId sw2: cluster) {
                        assertTrue(Arrays.binarySearch(expectedCluster, (int)sw2.getLong()) >= 0);
                        verifiedSwitches.add(sw2);
                    }
                }
            }
        }
    }
    protected void 
    verifyExpectedBroadcastPortsInClusters(int [][][] ebp) {
        verifyExpectedBroadcastPortsInClusters(ebp, true);
    }
    protected void 
    verifyExpectedBroadcastPortsInClusters(int [][][] ebp, 
                                           boolean tunnelsEnabled) {
        NodePortTuple npt = null;
        Set<NodePortTuple> expected = new HashSet<NodePortTuple>();
        for(int i=0; i<ebp.length; ++i) {
            int [][] nptList = ebp[i];
            expected.clear();
            for(int j=0; j<nptList.length; ++j) {
                npt = new NodePortTuple(DatapathId.of(nptList[j][0]), OFPort.of(nptList[j][1]));
                expected.add(npt);
            }
            TopologyInstance ti = topologyManager.getCurrentInstance(tunnelsEnabled);
            Set<NodePortTuple> computed = ti.getBroadcastNodePortsInCluster(npt.nodeId);
            log.info("computed: {}", computed);
            if (computed != null)
                assertTrue(computed.equals(expected));
            else if (computed == null)
                assertTrue(expected.isEmpty());
        }
    }
    public void createTopologyFromLinks(int [][] linkArray) throws Exception {
        ILinkDiscovery.LinkType type = ILinkDiscovery.LinkType.DIRECT_LINK;
        for (int i = 0; i < linkArray.length; i++) {
            int [] r = linkArray[i];
            if (r[4] == DIRECT_LINK)
                type= ILinkDiscovery.LinkType.DIRECT_LINK;
            else if (r[4] == MULTIHOP_LINK)
                type= ILinkDiscovery.LinkType.MULTIHOP_LINK;
            else if (r[4] == TUNNEL_LINK)
                type = ILinkDiscovery.LinkType.TUNNEL;
            topologyManager.addOrUpdateLink(DatapathId.of(r[0]), OFPort.of(r[1]), DatapathId.of(r[2]), OFPort.of(r[3]), U64.ZERO, type);
        }
        topologyManager.createNewInstance();
    }
    public TopologyManager getTopologyManager() {
        return topologyManager;
    }
    @Test
    public void testClusters() throws Exception {
        TopologyManager tm = getTopologyManager();
        {
            int [][] linkArray = { 
                                  {1, 1, 2, 1, DIRECT_LINK}, 
                                  {2, 2, 3, 2, DIRECT_LINK},
                                  {3, 1, 1, 2, DIRECT_LINK},
                                  {2, 3, 4, 2, DIRECT_LINK},
                                  {3, 3, 4, 1, DIRECT_LINK}
            };
            int [][] expectedClusters = {
                                         {1,2,3}, 
                                         {4}
            };
            createTopologyFromLinks(linkArray);
            verifyClusters(expectedClusters);
        }
        {
            int [][] linkArray = { 
                                  {5, 3, 6, 1, DIRECT_LINK} 
            };
            int [][] expectedClusters = {
                                         {1,2,3}, 
                                         {4},
                                         {5},
                                         {6}
            };
            createTopologyFromLinks(linkArray);
            verifyClusters(expectedClusters);
        }
        {
            int [][] linkArray = { 
                                  {6, 1, 5, 3, DIRECT_LINK} 
            };
            int [][] expectedClusters = {
                                         {1,2,3}, 
                                         {4},
                                         {5,6}
            };
            createTopologyFromLinks(linkArray);
            verifyClusters(expectedClusters);
        }
        {
            int [][] linkArray = { 
                                  {4, 2, 2, 3, DIRECT_LINK} 
            };
            int [][] expectedClusters = {
                                         {1,2,3,4},
                                         {5,6}
            };
            createTopologyFromLinks(linkArray);
            verifyClusters(expectedClusters);
        }
        {
            int [][] linkArray = { 
                                  {4, 3, 5, 1, DIRECT_LINK} 
            };
            int [][] expectedClusters = {
                                         {1,2,3,4},
                                         {5,6}
            };
            createTopologyFromLinks(linkArray);
            verifyClusters(expectedClusters);
        }
        {
            int [][] linkArray = { 
                                  {5, 2, 2, 4, DIRECT_LINK} 
            };
            int [][] expectedClusters = {
                                         {1,2,3,4,5,6}
            };
            createTopologyFromLinks(linkArray);
            verifyClusters(expectedClusters);
        }
        {
            int [][] linkArray = { 
                                  {3, 2, 2, 2, DIRECT_LINK}, 
                                  {2, 1, 1, 1, DIRECT_LINK},
                                  {1, 2, 3, 1, DIRECT_LINK},
                                  {4, 1, 3, 3, DIRECT_LINK},
                                  {5, 1, 4, 3, DIRECT_LINK},
                                  {2, 4, 5, 2, DIRECT_LINK}
            };
            int [][] expectedClusters = {
                                         {1,2,3,4,5,6}
            };
            createTopologyFromLinks(linkArray);
            verifyClusters(expectedClusters);
        }
        {
            tm.removeLink(DatapathId.of(5), OFPort.of((short)3), DatapathId.of(6), OFPort.of((short)1));
            tm.removeLink(DatapathId.of(6), OFPort.of((short)1), DatapathId.of(5), OFPort.of((short)3));
            int [][] expectedClusters = {
                                         {1,2,3,4,5},
            };
            topologyManager.createNewInstance();
            verifyClusters(expectedClusters);
        }
        {
            tm.removeSwitch(DatapathId.of(4));
            int [][] expectedClusters = {
                                         {1,2,3,5},
            };
            topologyManager.createNewInstance();
            verifyClusters(expectedClusters);
        }
    }
    @Test
    public void testLoopDetectionInSingleIsland() throws Exception {
        int [][] linkArray = {
                              {1, 1, 2, 1, DIRECT_LINK},
                              {2, 1, 1, 1, DIRECT_LINK},
                              {1, 2, 3, 1, DIRECT_LINK},
                              {3, 1, 1, 2, DIRECT_LINK},
                              {2, 2, 3, 2, DIRECT_LINK},
                              {3, 2, 2, 2, DIRECT_LINK},
                              {3, 3, 4, 1, DIRECT_LINK},
                              {4, 1, 3, 3, DIRECT_LINK},
                              {4, 2, 6, 2, DIRECT_LINK},
                              {6, 2, 4, 2, DIRECT_LINK},
                              {4, 3, 5, 1, DIRECT_LINK},
                              {5, 1, 4, 3, DIRECT_LINK},
                              {5, 2, 6, 1, DIRECT_LINK},
                              {6, 1, 5, 2, DIRECT_LINK},
        };
        int [][] expectedClusters = {
                                     {1, 2, 3, 4, 5, 6}
        };
        int [][][] expectedBroadcastPorts = {
                                             {{1,1}, {2,1}, {1,2}, {3,1}, {3,3}, {4,1}, {4,3}, {5,1}, {4,2}, {6,2}},
        };
        createTopologyFromLinks(linkArray);
        topologyManager.createNewInstance();
        verifyClusters(expectedClusters);
        verifyExpectedBroadcastPortsInClusters(expectedBroadcastPorts);
    }
    @Test
    public void testLoopDetectionWithIslands() throws Exception {
        {
            int [][] linkArray = {
                                  {1, 1, 2, 1, DIRECT_LINK},
                                  {2, 1, 1, 1, DIRECT_LINK},
                                  {1, 2, 3, 1, DIRECT_LINK},
                                  {3, 1, 1, 2, DIRECT_LINK},
                                  {2, 2, 3, 2, DIRECT_LINK},
                                  {3, 2, 2, 2, DIRECT_LINK},
                                  {4, 2, 6, 2, DIRECT_LINK},
                                  {6, 2, 4, 2, DIRECT_LINK},
                                  {4, 3, 5, 1, DIRECT_LINK},
                                  {5, 1, 4, 3, DIRECT_LINK},
                                  {5, 2, 6, 1, DIRECT_LINK},
                                  {6, 1, 5, 2, DIRECT_LINK},
            };
            int [][] expectedClusters = {
                                         {1, 2, 3}, 
                                         {4, 5, 6}
            };
            int [][][] expectedBroadcastPorts = {
                                                 {{1,1}, {2,1}, {1,2}, {3,1}},
                                                 {{4,3}, {5,1}, {4,2}, {6,2}},
            };
            createTopologyFromLinks(linkArray);
            topologyManager.createNewInstance();
            verifyClusters(expectedClusters);
            verifyExpectedBroadcastPortsInClusters(expectedBroadcastPorts);
        }
        {
            int [][] linkArray = {
                                  {3, 3, 4, 1, DIRECT_LINK},
                                  {4, 1, 3, 3, DIRECT_LINK},
            };
            int [][] expectedClusters = {
                                         {1, 2, 3, 4, 5, 6}
            };
            int [][][] expectedBroadcastPorts = {
                                                 {{1,1}, {2,1}, {1,2}, {3,1},
                                                  {3,3}, {4,1}, {4,3}, {5,1},
                                                  {4,2}, {6,2}},
            };
            createTopologyFromLinks(linkArray);
            topologyManager.createNewInstance();
            verifyClusters(expectedClusters, false);
            verifyExpectedBroadcastPortsInClusters(expectedBroadcastPorts);
        }
    }
    @Test
    public void testLinkRemovalOnBroadcastDomainPorts() throws Exception {
        {
            int [][] linkArray = {
                                  {1, 1, 2, 1, DIRECT_LINK},
                                  {2, 1, 1, 1, DIRECT_LINK},
                                  {1, 2, 3, 1, DIRECT_LINK},
                                  {3, 1, 1, 2, DIRECT_LINK},
                                  {2, 2, 3, 2, DIRECT_LINK},
                                  {3, 2, 2, 2, DIRECT_LINK},
                                  {1, 1, 3, 2, DIRECT_LINK},
            };
            int [][] expectedClusters = {
                                         {1, 3}, {2},
            };
            createTopologyFromLinks(linkArray);
            topologyManager.createNewInstance();
            if (topologyManager.getCurrentInstance() instanceof TopologyInstance)
                verifyClusters(expectedClusters);
        }
        {
            int [][] linkArray = {
                                  {1, 2, 3, 2, DIRECT_LINK},
            };
            int [][] expectedClusters = {
                                         {1}, {3}, {2},
            };
            createTopologyFromLinks(linkArray);
            topologyManager.createNewInstance();
            if (topologyManager.getCurrentInstance() instanceof TopologyInstance)
                verifyClusters(expectedClusters);
        }
    }
}
