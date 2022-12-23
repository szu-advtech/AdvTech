package net.floodlightcontroller.flowcache;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.util.OFMatchWithSwDpid;
import org.projectfloodlight.openflow.types.OFPort;
@Deprecated
public class OFMatchReconcile  {
    public enum ReconcileAction {
        DROP,
        NO_CHANGE,
        NEW_ENTRY,
        UPDATE_PATH,
        APP_INSTANCE_CHANGED,
        DELETE
    }
    public OFMatchWithSwDpid ofmWithSwDpid;
    public short priority;
    public byte action;
    public long cookie;
    public String appInstName;
    public String newAppInstName;
    public ReconcileAction rcAction;
    public OFPort outPort;
    public FloodlightContext cntx;
    public FlowReconcileQuery origReconcileQueryEvent;
    public OFMatchReconcile() {
        ofmWithSwDpid = new OFMatchWithSwDpid();
        rcAction = ReconcileAction.NO_CHANGE;
        cntx = new FloodlightContext();
    }
    public OFMatchReconcile(OFMatchReconcile copy) {
        ofmWithSwDpid =
            new OFMatchWithSwDpid(copy.ofmWithSwDpid.getMatch(),
                    copy.ofmWithSwDpid.getDpid());
        priority = copy.priority;
        action = copy.action;
        cookie = copy.cookie;
        appInstName = copy.appInstName;
        newAppInstName = copy.newAppInstName;
        rcAction = copy.rcAction;
        outPort = copy.outPort;
        cntx = new FloodlightContext();
        origReconcileQueryEvent = copy.origReconcileQueryEvent;
    }
    @Override
    public String toString() {
        return "OFMatchReconcile [" + ofmWithSwDpid + " priority=" + priority + " action=" + action + 
                " cookie=" + cookie + " appInstName=" + appInstName + " newAppInstName=" + newAppInstName + 
                " ReconcileAction=" + rcAction + "outPort=" + outPort + "]";
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
                 + ((appInstName == null) ? 0 : appInstName.hashCode());
        result = prime
                 + ((newAppInstName == null) ? 0 : newAppInstName.hashCode());
                 + ((ofmWithSwDpid == null) ? 0 : ofmWithSwDpid.hashCode());
                 + ((rcAction == null) ? 0 : rcAction.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (!(obj instanceof OFMatchReconcile)) {
            return false;
        }
        OFMatchReconcile other = (OFMatchReconcile) obj;
        if (action != other.action) {
            return false;
        }
        if (appInstName == null) {
            if (other.appInstName != null) {
                return false;
            }
        } else if (!appInstName.equals(other.appInstName)) {
            return false;
        }
        if (cookie != other.cookie) {
            return false;
        }
        if (newAppInstName == null) {
            if (other.newAppInstName != null) {
                return false;
            }
        } else if (!newAppInstName.equals(other.newAppInstName)) {
            return false;
        }
        if (ofmWithSwDpid == null) {
            if (other.ofmWithSwDpid != null) {
                return false;
            }
        } else if (!ofmWithSwDpid.equals(other.ofmWithSwDpid)) {
            return false;
        }
        if (outPort != other.outPort) {
            return false;
        }
        if (priority != other.priority) {
            return false;
        }
        if (rcAction != other.rcAction) {
            return false;
        }
        return true;
    }
}