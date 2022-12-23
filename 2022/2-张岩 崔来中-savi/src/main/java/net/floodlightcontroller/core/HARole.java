package net.floodlightcontroller.core;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public enum HARole {
    ACTIVE(OFControllerRole.ROLE_MASTER),
    STANDBY(OFControllerRole.ROLE_SLAVE);
    private static final Logger logger = LoggerFactory.getLogger(HARole.class);
    private final OFControllerRole ofRole;
    HARole(OFControllerRole ofRole) {
        this.ofRole = ofRole;
    }
    public static HARole valueOfBackwardsCompatible(String roleString) throws IllegalArgumentException {
        roleString = roleString.trim().toUpperCase();
        if("MASTER".equals(roleString)) {
            logger.warn("got legacy role name MASTER - normalized to ACTIVE", roleString);
            if(logger.isDebugEnabled()) {
               logger.debug("Legacy role call stack", new IllegalArgumentException());
            }
            roleString = "ACTIVE";
        } else if ("SLAVE".equals(roleString)) {
            logger.warn("got legacy role name SLAVE - normalized to STANDBY", roleString);
            if(logger.isDebugEnabled()) {
               logger.debug("Legacy role call stack", new IllegalArgumentException());
            }
            roleString = "STANDBY";
        }
        return valueOf(roleString);
    }
    public OFControllerRole getOFRole() {
        return ofRole;
    }
    public static HARole ofOFRole(OFControllerRole role) {
        switch(role) {
            case ROLE_MASTER:
            case ROLE_EQUAL:
                return ACTIVE;
            case ROLE_SLAVE:
                return STANDBY;
            default:
                throw new IllegalArgumentException("Unmappable controller role: " + role);
        }
    }
}
