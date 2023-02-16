package net.floodlightcontroller.util;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public enum BundleAction {
    START,
    STOP,
    UNINSTALL,
    REFRESH;
    public static List<BundleAction> getAvailableActions(BundleState state) {
        List<BundleAction> actions = new ArrayList<BundleAction>();
        if (Arrays.binarySearch(new BundleState[] {
                BundleState.ACTIVE, BundleState.STARTING,
                BundleState.UNINSTALLED }, state) < 0) {
            actions.add(START);
        }
        if (Arrays.binarySearch(new BundleState[] {
                BundleState.ACTIVE}, state) >= 0) {
            actions.add(STOP);
        }
        if (Arrays.binarySearch(new BundleState[] {
                BundleState.UNINSTALLED}, state) < 0) {
            actions.add(UNINSTALL);
        }
        actions.add(REFRESH);
        return actions;
    }
}
