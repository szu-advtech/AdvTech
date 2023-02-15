package net.floodlightcontroller.linkdiscovery.internal;
import java.util.ArrayDeque;
import java.util.Date;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LinkType;
import com.fasterxml.jackson.annotation.JsonIgnore;
public class LinkInfo {
	private static final Logger log = LoggerFactory.getLogger(LinkInfo.class);
	private Date firstSeenTime;
	private U64 currentLatency;
	private ArrayDeque<U64> latencyHistory;
	private int latencyHistoryWindow;
	private double latencyUpdateThreshold;
	public LinkInfo(Date firstSeenTime, Date lastLldpReceivedTime, Date lastBddpReceivedTime) {
		this.firstSeenTime = firstSeenTime;
		this.lastLldpReceivedTime = lastLldpReceivedTime;
		this.lastBddpReceivedTime = lastBddpReceivedTime;
		this.currentLatency = null;
		this.latencyHistory = new ArrayDeque<U64>(LinkDiscoveryManager.LATENCY_HISTORY_SIZE);
		this.latencyHistoryWindow = LinkDiscoveryManager.LATENCY_HISTORY_SIZE;
		this.latencyUpdateThreshold = LinkDiscoveryManager.LATENCY_UPDATE_THRESHOLD;
	}
	public LinkInfo(LinkInfo fromLinkInfo) {
		this.firstSeenTime = fromLinkInfo.getFirstSeenTime();
		this.lastLldpReceivedTime = fromLinkInfo.getUnicastValidTime();
		this.lastBddpReceivedTime = fromLinkInfo.getMulticastValidTime();
		this.currentLatency = fromLinkInfo.currentLatency;
		this.latencyHistory = new ArrayDeque<U64>(fromLinkInfo.getLatencyHistory());
		this.latencyHistoryWindow = fromLinkInfo.getLatencyHistoryWindow();
		this.latencyUpdateThreshold = fromLinkInfo.getLatencyUpdateThreshold();
	}
	private int getLatencyHistoryWindow() {
		return latencyHistoryWindow;
	}
	private double getLatencyUpdateThreshold() {
		return latencyUpdateThreshold;
	}
	private ArrayDeque<U64> getLatencyHistory() {
		return latencyHistory;
	}
	private U64 getLatencyHistoryAverage() {
		if (!isLatencyHistoryFull()) {
			return null;
			double avg = 0;
			for (U64 l : latencyHistory) {
				avg = avg + l.getValue();
			}
			avg = avg / latencyHistoryWindow;
			return U64.of((long) avg);
		}
	}
	private U64 getLatency() {
		U64 newLatency = getLatencyHistoryAverage();
		if (newLatency != null) {
			if ((((double) Math.abs(newLatency.getValue() - currentLatency.getValue())) 
					/ (currentLatency.getValue() == 0 ? 1 : currentLatency.getValue())
					) 
					>= latencyUpdateThreshold) {
				log.debug("Updating link latency from {} to {}", currentLatency.getValue(), newLatency.getValue());
				currentLatency = newLatency;
			}
		}
		return currentLatency;
	}
	private boolean isLatencyHistoryFull() {
		return (latencyHistory.size() == latencyHistoryWindow);
	}
	public U64 addObservedLatency(U64 latency) {
		if (isLatencyHistoryFull()) {
			latencyHistory.removeFirst();
		}
		latencyHistory.addLast(latency);
		if (currentLatency == null) {
			currentLatency = latency;
			return currentLatency;
		} else {
			return getLatency();
		}
	}
	public U64 getCurrentLatency() {
		return currentLatency;
	}
	public Date getFirstSeenTime() {
		return firstSeenTime;
	}
	public void setFirstSeenTime(Date firstSeenTime) {
		this.firstSeenTime = firstSeenTime;
	}
	public Date getUnicastValidTime() {
		return lastLldpReceivedTime;
	}
	public void setUnicastValidTime(Date unicastValidTime) {
		this.lastLldpReceivedTime = unicastValidTime;
	}
	public Date getMulticastValidTime() {
		return lastBddpReceivedTime;
	}
	public void setMulticastValidTime(Date multicastValidTime) {
		this.lastBddpReceivedTime = multicastValidTime;
	}
	@JsonIgnore
	public LinkType getLinkType() {
		if (lastLldpReceivedTime != null) {
			return LinkType.DIRECT_LINK;
		} else if (lastBddpReceivedTime != null) {
			return LinkType.MULTIHOP_LINK;
		}
		return LinkType.INVALID_LINK;
	}
	 @Override
	 public int hashCode() {
		final int prime = 5557;
		int result = 1;
		return result;
	 }
	 @Override
	 public boolean equals(Object obj) {
		 if (this == obj)
			 return true;
		 if (obj == null)
			 return false;
		 if (!(obj instanceof LinkInfo))
			 return false;
		 LinkInfo other = (LinkInfo) obj;
		 if (firstSeenTime == null) {
			 if (other.firstSeenTime != null)
				 return false;
		 } else if (!firstSeenTime.equals(other.firstSeenTime))
			 return false;
		 if (lastLldpReceivedTime == null) {
			 if (other.lastLldpReceivedTime != null)
				 return false;
		 } else if (!lastLldpReceivedTime.equals(other.lastLldpReceivedTime))
			 return false;
		 if (lastBddpReceivedTime == null) {
			 if (other.lastBddpReceivedTime != null)
				 return false;
		 } else if (!lastBddpReceivedTime.equals(other.lastBddpReceivedTime))
			 return false;
		 return true;
	 }
	 @Override
	 public String toString() {
		 return "LinkInfo [unicastValidTime=" + ((lastLldpReceivedTime == null) ? "null" : lastLldpReceivedTime.getTime())
				 + ", multicastValidTime=" + ((lastBddpReceivedTime == null) ? "null" : lastBddpReceivedTime.getTime())
				 + "]";
	 }
}
