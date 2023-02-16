package net.floodlightcontroller.core;
import java.util.Date;
import com.fasterxml.jackson.annotation.JsonProperty;
public class RoleInfo {
    private final HARole role;
    private final String roleChangeDescription;
    private final Date roleChangeDateTime;
    public RoleInfo(HARole role, String description, Date dt) {
        this.role = role;
        this.roleChangeDescription = description;
        this.roleChangeDateTime = dt;
    }
    public HARole getRole() {
        return role;
    }
    @JsonProperty(value="change-description")
    public String getRoleChangeDescription() {
        return roleChangeDescription;
    }
    @JsonProperty(value="change-date-time")
    public Date getRoleChangeDateTime() {
        return roleChangeDateTime;
    }
}
