package net.floodlightcontroller.savi.module;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.ICMPv6;
import net.floodlightcontroller.packet.IPv6;
import net.floodlightcontroller.routing.IRoutingDecision.RoutingAction;
import net.floodlightcontroller.savi.action.Action;
import net.floodlightcontroller.savi.action.Action.ActionFactory;
import net.floodlightcontroller.savi.action.ClearIPv6BindingAction;
import net.floodlightcontroller.savi.action.ClearPortBindingAction;
import net.floodlightcontroller.savi.action.ClearSwitchBindingAction;
import net.floodlightcontroller.savi.binding.Binding;
import net.floodlightcontroller.savi.binding.BindingPool;
import net.floodlightcontroller.savi.binding.BindingStatus;
public class SLAACService extends SAVIBaseService {
	protected static int UPDATE_DELAY = 2;
	BindingPool<IPv6Address> pool;
	List<Binding<IPv6Address>> updateQueue;
	protected SingletonTask timer;
	@Override
	public void startUpService() {
		pool = new BindingPool<>();
		updateQueue = new  CopyOnWriteArrayList<>();
		ScheduledExecutorService ses = threadPoolService.getScheduledExecutor();
		timer = new SingletonTask(ses, new Runnable() {
			@Override
			public void run() {
				int tmp = (int)(System.currentTimeMillis()/1000);
				synchronized (updateQueue) {
					for(Binding<IPv6Address> entry:updateQueue){
						if((tmp-entry.getBindingTime())>UPDATE_DELAY){
							entry.setStatus(BindingStatus.BOUND);
							entry.setBindingTime();
							List<Action> actions = new ArrayList<>();
							actions.add(ActionFactory.getBindIPv6Action(entry));
							List<FlowAction> flowActions=new ArrayList<>();
							Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
							mb.setExact(MatchField.ETH_SRC, entry.getMacAddress());
							mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
							mb.setExact(MatchField.IPV6_SRC, (IPv6Address)entry.getAddress());
							mb.setExact(MatchField.IN_PORT, entry.getSwitchPort().getPort());
							List<OFInstruction> instructions = new ArrayList<>();
							instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(TableId.of(1)));
							flowActions.add(FlowActionFactory.getFlowAddAction(
									entry.getSwitchPort().getSwitchDPID(), 
									TableId.of(0), mb.build(), null, instructions, 3));
							saviProvider.pushActions(actions);
							updateQueue.remove(entry);
						}
					}
				}
				timer.reschedule(TIMER_DELAY, TimeUnit.SECONDS);
			}
		});
		timer.reschedule(100, TimeUnit.MILLISECONDS);
	}
	protected RoutingAction processICMPv6(SwitchPort switchPort,Ethernet eth){
		IPv6 ipv6 = (IPv6)eth.getPayload();
		ICMPv6 icmpv6 = (ICMPv6)ipv6.getPayload();
		switch(icmpv6.getICMPv6Type()){
		case ICMPv6.NEIGHBOR_ADVERTISEMENT:
			return processNeighborAdvertisement(switchPort, eth);
		case ICMPv6.NEIGHBOR_SOLICITATION:
			return processNeighborSoliciation(switchPort, eth);
		case ICMPv6.ROUTER_ADVERTSEMENT:
			return processRouterAdvertisement(switchPort, eth);
		case ICMPv6.ROUTER_SOLICITATION:
			return processRouterSolification(switchPort, eth);
		}
		return null;
	}
	protected boolean isICMPv6(IPv6 ipv6){
		if(ipv6.getNextHeader().equals(IpProtocol.IPv6_ICMP)){
			return true;
		}
		return false;
	}
	protected RoutingAction processNeighborAdvertisement(SwitchPort switchPort,Ethernet eth){
		IPv6 ipv6 = (IPv6)eth.getPayload();
		ICMPv6 icmpv6 = (ICMPv6)ipv6.getPayload();
		MacAddress macAddress = eth.getSourceMACAddress();
		IPv6Address targetAddress = icmpv6.getTargetAddress();
		Binding<IPv6Address> binding = null;
		if(pool.isContain(targetAddress)){
			binding = pool.getBinding(targetAddress);
			if(binding.check(macAddress, targetAddress)){
				return RoutingAction.FORWARD_OR_FLOOD;
			}
		}
		return RoutingAction.NONE;
	}
	protected RoutingAction processNeighborSoliciation(SwitchPort switchPort,Ethernet eth){
		List<Action> actions = new ArrayList<>();
		IPv6 ipv6 = (IPv6)eth.getPayload();
		ICMPv6 icmpv6 = (ICMPv6)ipv6.getPayload();
		MacAddress macAddress = eth.getSourceMACAddress();
		IPv6Address targetAddress = icmpv6.getTargetAddress();
		Binding<IPv6Address> binding = null;
			return RoutingAction.NONE;
		}
		if(!pool.isContain(macAddress)){
			pool.addHardwareBinding(macAddress, switchPort);
		}
		actions.add(ActionFactory.getCheckIPv6Binding(switchPort, macAddress, targetAddress));
		if(saviProvider.pushActions(actions)){	
			return RoutingAction.FORWARD_OR_FLOOD;
		}
		else if(pool.isContain(targetAddress)){
			actions.clear();
			binding = pool.getBinding(targetAddress);
			if(binding.check(macAddress,targetAddress)){
				actions.add(ActionFactory.getFloodAction(switchPort.getSwitchDPID(), switchPort.getPort(), eth));
				saviProvider.pushActions(actions);
			}
		}
		else{
			actions.clear();
			binding = new Binding<>();
			binding.setAddress(targetAddress);
			binding.setMacAddress(macAddress);
			binding.setStatus(BindingStatus.DETECTING);
			binding.setBindingTime();
			binding.setSwitchPort(switchPort);
			pool.addBinding(targetAddress, binding);
			synchronized(updateQueue){
				updateQueue.add(binding);
			}
			actions.add(ActionFactory.getFloodAction(switchPort.getSwitchDPID(), switchPort.getPort(), eth));
			saviProvider.pushActions(actions);
		}
		return RoutingAction.NONE;
	}
	protected RoutingAction processRouterAdvertisement(SwitchPort switchPort,Ethernet eth){
		return null;
	}
	protected RoutingAction processRouterSolification(SwitchPort switchPort,Ethernet eth){
		return null;
	}
	@Override
	public boolean match(Ethernet eth) {
		if(eth.getEtherType() == EthType.IPv6){
			IPv6 ipv6 = (IPv6)eth.getPayload();
			if(ipv6.getNextHeader().equals(IpProtocol.IPv6_ICMP)){
				ICMPv6 icmpv6 = (ICMPv6)ipv6.getPayload();
				byte type = icmpv6.getICMPv6Type();
				if(type == ICMPv6.NEIGHBOR_SOLICITATION){
					if(ipv6.getSourceAddress().isUnspecified()){
						return true;
					}
				}
				if(type == ICMPv6.NEIGHBOR_ADVERTISEMENT){
					if(ipv6.getDestinationAddress().isBroadcast()){
						return true;
					}
				}
			}
		}
		return false;
	}
	@Override
	protected void doClearIPv6BindingAction(ClearIPv6BindingAction action){
		pool.delBinding(action.getIpv6Address());
	}
	@Override
	protected void doClearPortBindingAction(ClearPortBindingAction action){
	}
	@Override
	protected void doClearSwitchBindingAction(ClearSwitchBindingAction action){
		pool.delSwitch(action.getSwitchId());
	}
	@Override
	public List<Match> getMatches() {
		List<Match> array = new ArrayList<>();
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
		mb.setExact(MatchField.IP_PROTO, IpProtocol.IPv6_ICMP);
		array.add(mb.build());
		return array;
	}
	@Override
	public RoutingAction process(SwitchPort switchPort, Ethernet eth) {
		return processICMPv6(switchPort, eth);
	}
	@Override
	public void handlePortDown(SwitchPort switchPort) {
	}
	@Override
	public void checkDeadline(){
		List<Action> actions = new ArrayList<>();
		for(Binding<IPv6Address> binding:pool.getAllBindings()){
			if(binding.isLeaseExpired()){
				actions.add(ActionFactory.getUnbindIPv6Action(binding.getAddress(), binding));
				pool.delBinding(binding.getAddress());
			}
		}
		if(actions.size()>0){
			saviProvider.pushActions(actions);
		}
	}
}
