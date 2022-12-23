package net.floodlightcontroller.learningswitch;
import java.util.Map;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.types.MacVlanPair;
public interface ILearningSwitchService extends IFloodlightService {
    public Map<IOFSwitch, Map<MacVlanPair, OFPort>> getTable();
}
