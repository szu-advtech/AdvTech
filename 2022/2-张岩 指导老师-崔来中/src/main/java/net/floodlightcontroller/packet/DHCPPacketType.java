package net.floodlightcontroller.packet;
public enum DHCPPacketType {
    DHCPDISCOVER        (1),
    DHCPOFFER           (2),
    DHCPREQUEST         (3),
    DHCPDECLINE         (4),
    DHCPACK             (5),
    DHCPNAK             (6),
    DHCPRELEASE         (7),
    DHCPINFORM          (8),
    DHCPFORCERENEW      (9),
    DHCPLEASEQUERY      (10),
    DHCPLEASEUNASSIGNED (11),
    DHCPLEASEUNKNOWN    (12),
    DHCPLEASEACTIVE     (13);
    protected int value;
    private DHCPPacketType(int value) {
        this.value = value;
    }
    public int getValue() {
        return value;
    }
    public String toString(){
        switch (value) {
            case 1:
                return "DHCPDISCOVER";
            case 2:
                return "DHCPOFFER";
            case 3:
                return "DHCPREQUEST";
            case 4:
                return "DHCPDECLINE";
            case 5:
                return "DHCPACK";
            case 6:
                return "DHCPNAK";
            case 7:
                return "DHCPRELEASE";
            case 8:
                return "DHCPINFORM";
            case 9:
                return "DHCPFORCERENEW";
            case 10:
                return "DHCPLEASEQUERY";
            case 11:
                return "DHCPLEASEUNASSIGNED";
            case 12:
                return "DHCPLEASEUNKNOWN";
            case 13:
                return "DHCPLEASEACTIVE";
            default:
                return "UNKNOWN";
        }
    }
    public static DHCPPacketType getType(int value) {
        switch (value) {
            case 1:
                return DHCPDISCOVER;
            case 2:
                return DHCPOFFER;
            case 3:
                return DHCPREQUEST;
            case 4:
                return DHCPDECLINE;
            case 5:
                return DHCPACK;
            case 6:
                return DHCPNAK;
            case 7:
                return DHCPRELEASE;
            case 8:
                return DHCPINFORM;
            case 9:
                return DHCPFORCERENEW;
            case 10:
                return DHCPLEASEQUERY;
            case 11:
                return DHCPLEASEUNASSIGNED;
            case 12:
                return DHCPLEASEUNKNOWN;
            case 13:
                return DHCPLEASEACTIVE;
        }
        return null;
    }
}
