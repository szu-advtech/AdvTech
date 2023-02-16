package net.floodlightcontroller.util;
public enum BundleState {
    ACTIVE              (32),
    INSTALLED           (2),
    RESOLVED            (4),
    STARTING            (8),
    STOPPING            (16),
    UNINSTALLED         (1);
    protected int value;
    private BundleState(int value) {
        this.value = value;
    }
    public int getValue() {
        return value;
    }
    public static BundleState getState(int value) {
        switch (value) {
            case 32:
                return ACTIVE;
            case 2:
                return INSTALLED;
            case 4:
                return RESOLVED;
            case 8:
                return STARTING;
            case 16:
                return STOPPING;
            case 1:
                return UNINSTALLED;
        }
        return null;
    }
}
