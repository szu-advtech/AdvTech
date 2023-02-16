package net.floodlightcontroller.loadbalancer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
@JsonSerialize(using=LBMemberSerializer.class)
public class LBMember {
    protected String id;
    protected int address;
    protected short port;
    protected String macString;
    protected int connectionLimit;
    protected short adminState;
    protected short status;
    protected String poolId;
    protected String vipId;
    public LBMember() {
        address = 0;
        macString = null;
        port = 0;
        connectionLimit = 0;
        adminState = 0;
        status = 0;
        poolId = null;
        vipId = null;
    }
}
