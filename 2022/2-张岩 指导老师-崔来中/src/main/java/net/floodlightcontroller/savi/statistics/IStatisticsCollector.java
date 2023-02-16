package net.floodlightcontroller.savi.statistics;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.savi.statistics.web.KeyValuePairSet;
public interface IStatisticsCollector extends IFloodlightService {
	public KeyValuePairSet  getStatistics();
}
