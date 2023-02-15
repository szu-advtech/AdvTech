package org.sdnplatform.sync.internal.config;
import org.sdnplatform.sync.error.SyncException;
import org.sdnplatform.sync.internal.SyncManager;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
public interface IClusterConfigProvider {
    public void init(SyncManager syncManager,
                     FloodlightModuleContext context) throws SyncException;
    public ClusterConfig getConfig() throws SyncException;
}
