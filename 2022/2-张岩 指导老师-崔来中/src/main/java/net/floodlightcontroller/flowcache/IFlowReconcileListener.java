package net.floodlightcontroller.flowcache;
import java.util.ArrayList;
import net.floodlightcontroller.core.IListener;
import org.projectfloodlight.openflow.protocol.OFType;
@Deprecated
public interface IFlowReconcileListener extends IListener<OFType> {
    public Command reconcileFlows(ArrayList<OFMatchReconcile> ofmRcList);
}
