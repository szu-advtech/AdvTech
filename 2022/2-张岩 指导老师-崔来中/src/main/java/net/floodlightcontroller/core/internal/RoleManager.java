package net.floodlightcontroller.core.internal;
import java.util.Map.Entry;
import javax.annotation.Nonnull;
import java.util.Date;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IHAListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.IShutdownService;
import net.floodlightcontroller.core.RoleInfo;
import net.floodlightcontroller.core.internal.Controller.IUpdate;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.types.DatapathId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.base.Preconditions;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
public class RoleManager {
    private volatile RoleInfo currentRoleInfo;
    private final Controller controller;
    private final IShutdownService shutdownService;
    private final RoleManagerCounters counters;
    private static final Logger log =
            LoggerFactory.getLogger(RoleManager.class);
    public RoleManager(@Nonnull Controller controller,
            @Nonnull IShutdownService shutdownService,
            @Nonnull HARole role,
            @Nonnull String roleChangeDescription) {
        Preconditions.checkNotNull(controller, "controller must not be null");
        Preconditions.checkNotNull(role, "role must not be null");
        Preconditions.checkNotNull(roleChangeDescription, "roleChangeDescription must not be null");
        Preconditions.checkNotNull(shutdownService, "shutdownService must not be null");
        this.currentRoleInfo = new RoleInfo(role,
                                       roleChangeDescription,
                                       new Date());
        this.controller = controller;
        this.shutdownService = shutdownService;
        this.counters = new RoleManagerCounters(controller.getDebugCounter());
    }
    public synchronized void reassertRole(OFSwitchHandshakeHandler ofSwitchHandshakeHandler, HARole role) {
        if (this.getRole() != role)
            return;
        ofSwitchHandshakeHandler.sendRoleRequestIfNotPending(this.getRole().getOFRole());
    }
    public synchronized void setRole(@Nonnull HARole role, @Nonnull String roleChangeDescription) {
        Preconditions.checkNotNull(role, "role must not be null");
        Preconditions.checkNotNull(roleChangeDescription, "roleChangeDescription must not be null");
        if (role == getRole()) {
            counters.setSameRole.increment();
            log.debug("Received role request for {} but controller is "
                    + "already {}. Ignoring it.", role, this.getRole());
            return;
        }
        if (this.getRole() == HARole.STANDBY && role == HARole.ACTIVE) {
            counters.setRoleMaster.increment();
        }
        log.info("Received role request for {} (reason: {})."
                + " Initiating transition", role, roleChangeDescription);
        currentRoleInfo =
                new RoleInfo(role, roleChangeDescription, new Date());
        controller.addUpdateToQueue(new HARoleUpdate(role));
        controller.addUpdateToQueue(new SwitchRoleUpdate(role));
    }
    @SuppressFBWarnings(value="UG_SYNC_SET_UNSYNC_GET",
                        justification = "setter is synchronized for mutual exclusion, "
                                + "currentRoleInfo is volatile, so no sync on getter needed")
    public synchronized HARole getRole() {
        return currentRoleInfo.getRole();
    }
    public synchronized OFControllerRole getOFControllerRole() {
        return getRole().getOFRole();
    }
    public RoleInfo getRoleInfo() {
        return currentRoleInfo;
    }
    private void attemptActiveTransition() {
         if(!switchesHaveAnotherMaster()){
             setRole(HARole.ACTIVE, "Leader election assigned ACTIVE role");
         }
     }
    private boolean switchesHaveAnotherMaster() {
        IOFSwitchService switchService = controller.getSwitchService();
        for(Entry<DatapathId, IOFSwitch> switchMap : switchService.getAllSwitchMap().entrySet()){
            IOFSwitchBackend sw = (IOFSwitchBackend) switchMap.getValue();
            if(sw.hasAnotherMaster()){
                return true;
            }
        }
        return false;
    }
    public void notifyControllerConnectionUpdate() {
        if(currentRoleInfo.getRole() != HARole.ACTIVE) {
            attemptActiveTransition();
        }
    }
    private class HARoleUpdate implements IUpdate {
        private final HARole newRole;
        public HARoleUpdate(HARole newRole) {
            this.newRole = newRole;
        }
        @Override
        public void dispatch() {
            if (log.isDebugEnabled()) {
                log.debug("Dispatching HA Role update newRole = {}",
                          newRole);
            }
            for (IHAListener listener : controller.haListeners.getOrderedListeners()) {
                if (log.isTraceEnabled()) {
                    log.trace("Calling HAListener {} with transitionTo{}",
                              listener.getName(), newRole);
                }
                switch(newRole) {
                    case ACTIVE:
                        listener.transitionToActive();
                        break;
                    case STANDBY:
                        listener.transitionToStandby();
                        break;
                }
           }
           controller.setNotifiedRole(newRole);
           if(newRole == HARole.STANDBY) {
               String reason = String.format("Received role request to "
                       + "transition from ACTIVE to STANDBY (reason: %s)",
                       getRoleInfo().getRoleChangeDescription());
               shutdownService.terminate(reason, 0);
           }
        }
    }
    public class SwitchRoleUpdate implements IUpdate {
        private final HARole role;
        public SwitchRoleUpdate(HARole role) {
            this.role = role;
        }
        @Override
        public void dispatch() {
            if (log.isDebugEnabled()) {
                log.debug("Dispatching switch role update newRole = {}, switch role = {}",
                          this.role, this.role.getOFRole());
            }
            for (OFSwitchHandshakeHandler h: controller.getSwitchService().getSwitchHandshakeHandlers())
                h.sendRoleRequest(this.role.getOFRole());
        }
    }
    public RoleManagerCounters getCounters() {
        return this.counters;
    }
}
