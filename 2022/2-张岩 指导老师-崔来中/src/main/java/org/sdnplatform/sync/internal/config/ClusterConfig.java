package org.sdnplatform.sync.internal.config;
import java.security.KeyStore;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import org.sdnplatform.sync.error.SyncException;
public class ClusterConfig {
    public static final short NODE_ID_UNCONFIGURED = Short.MAX_VALUE;
    private HashMap<Short, Node> allNodes =
            new HashMap<Short, Node>();
    private HashMap<Short, List<Node>> localDomains =
            new HashMap<Short, List<Node>>();
    private Node thisNode;
    private AuthScheme authScheme;
    private String keyStorePath;
    private String keyStorePassword;
    private String listenAddress;
    public ClusterConfig() {
        super();
    }
    public ClusterConfig(List<Node> nodes, short thisNodeId)
            throws SyncException {
        init(nodes, thisNodeId, AuthScheme.NO_AUTH, null, null);
    }
    public ClusterConfig(List<Node> nodes, short thisNodeId,
                         AuthScheme authScheme,
                         String keyStorePath, 
                         String keyStorePassword)
            throws SyncException {
        init(nodes, thisNodeId, authScheme, keyStorePath, keyStorePassword);
    }
    public ClusterConfig(List<Node> nodes, short thisNodeId,
                         String listenAddress,
                         AuthScheme authScheme,
                         String keyStorePath, 
                         String keyStorePassword)
            throws SyncException {
        init(nodes, thisNodeId, authScheme, keyStorePath, keyStorePassword);
        this.listenAddress = listenAddress;
    }
    public Collection<Node> getNodes() {
        return Collections.unmodifiableCollection(allNodes.values());
    }
    public Collection<Node> getDomainNodes() {
        return getDomainNodes(thisNode.getDomainId());
    }
    public Collection<Node> getDomainNodes(short domainId) {
        List<Node> r = localDomains.get(domainId);
        return Collections.unmodifiableCollection(r);
    }
    public Node getNode() {
        return thisNode;
    }
    public Node getNode(short nodeId) {
        return allNodes.get(nodeId);
    }
    public String getListenAddress() {
        return listenAddress;
    }
    public AuthScheme getAuthScheme() {
        return authScheme;
    }
    public String getKeyStorePath() {
        return keyStorePath;
    }
    public String getKeyStorePassword() {
        return keyStorePassword;
    }
    private void addNode(Node node) throws SyncException {
        Short nodeId = node.getNodeId();
        if (allNodes.get(nodeId) != null) {
            throw new SyncException("Error adding node " + node +
                    ": a node with that ID already exists");
        }
        allNodes.put(nodeId, node);
        Short domainId = node.getDomainId();
        List<Node> localDomain = localDomains.get(domainId);
        if (localDomain == null) {
            localDomains.put(domainId,
                             localDomain = new ArrayList<Node>());
        }
        localDomain.add(node);
    }
    private void init(List<Node> nodes, short thisNodeId,
                      AuthScheme authScheme,
                      String keyStorePath, 
                      String keyStorePassword)
            throws SyncException {
        for (Node n : nodes) {
            addNode(n);
        }
        thisNode = getNode(thisNodeId);
        if (thisNode == null) {
            throw new SyncException("Cannot set thisNode " +
                    "node: No node with ID " + thisNodeId);
        }
        this.authScheme = authScheme;
        if (this.authScheme == null) 
            this.authScheme = AuthScheme.NO_AUTH;
        this.keyStorePath = keyStorePath;
        this.keyStorePassword = keyStorePassword;
    }
    @Override
    public String toString() {
        return "ClusterConfig [allNodes=" + allNodes + ", authScheme="
               + authScheme + ", keyStorePath=" + keyStorePath
               + ", keyStorePassword is " + 
               (keyStorePassword == null ? "unset" : "set") + "]";
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
                 + ((allNodes == null) ? 0 : allNodes.hashCode());
                 + ((authScheme == null) ? 0 : authScheme.hashCode());
        result = prime
                 + ((keyStorePassword == null) ? 0
                                              : keyStorePassword.hashCode());
                 + ((keyStorePath == null) ? 0 : keyStorePath.hashCode());
                 + ((localDomains == null) ? 0 : localDomains.hashCode());
                 + ((thisNode == null) ? 0 : thisNode.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        ClusterConfig other = (ClusterConfig) obj;
        if (allNodes == null) {
            if (other.allNodes != null) return false;
        } else if (!allNodes.equals(other.allNodes)) return false;
        if (authScheme != other.authScheme) return false;
        if (keyStorePassword == null) {
            if (other.keyStorePassword != null) return false;
        } else if (!keyStorePassword.equals(other.keyStorePassword))
                                                                    return false;
        if (keyStorePath == null) {
            if (other.keyStorePath != null) return false;
        } else if (!keyStorePath.equals(other.keyStorePath)) return false;
        if (thisNode == null) {
            if (other.thisNode != null) return false;
        } else if (!thisNode.equals(other.thisNode)) return false;
        return true;
    }
}
