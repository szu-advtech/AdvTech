package org.sdnplatform.sync.internal.version;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.sdnplatform.sync.IVersion;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.Lists;
public class VectorClock implements IVersion, Serializable, Cloneable {
    private static final long serialVersionUID = 7663945747147638702L;
    private static final int MAX_NUMBER_OF_VERSIONS = Short.MAX_VALUE;
    private final List<ClockEntry> versions;
    private final long timestamp;
    public VectorClock() {
        this(new ArrayList<ClockEntry>(0), System.currentTimeMillis());
    }
    public VectorClock(long timestamp) {
        this(new ArrayList<ClockEntry>(0), timestamp);
    }
    @JsonCreator
    public VectorClock(@JsonProperty("entries") List<ClockEntry> versions, 
                       @JsonProperty("timestamp") long timestamp) {
        this.versions = versions;
        this.timestamp = timestamp;
    }
    public VectorClock incremented(int nodeId, long time) {
        if(nodeId < 0 || nodeId > Short.MAX_VALUE)
            throw new IllegalArgumentException(nodeId
                                               + " is outside the acceptable range of node ids.");
        List<ClockEntry> newversions = Lists.newArrayList(versions);
        boolean found = false;
        int index = 0;
        for(; index < newversions.size(); index++) {
            if(newversions.get(index).getNodeId() == nodeId) {
                found = true;
                break;
            } else if(newversions.get(index).getNodeId() > nodeId) {
                found = false;
                break;
            }
        }
        if(found) {
            newversions.set(index, newversions.get(index).incremented());
        } else if(index < newversions.size() - 1) {
            newversions.add(index, new ClockEntry((short) nodeId, 1));
        } else {
            if(newversions.size() > MAX_NUMBER_OF_VERSIONS)
                throw new IllegalStateException("Vector clock is full!");
            newversions.add(index, new ClockEntry((short) nodeId, 1));
        }
        return new VectorClock(newversions, time);
    }
    @Override
    public VectorClock clone() {
        return new VectorClock(Lists.newArrayList(versions), this.timestamp);
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
                 + ((versions == null) ? 0 : versions.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        VectorClock other = (VectorClock) obj;
        if (timestamp != other.timestamp) return false;
        if (versions == null) {
            if (other.versions != null) return false;
        } else if (!versions.equals(other.versions)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("version(");
        if(this.versions.size() > 0) {
            for(int i = 0; i < this.versions.size() - 1; i++) {
                builder.append(this.versions.get(i));
                builder.append(", ");
            }
            builder.append(this.versions.get(this.versions.size() - 1));
        }
        builder.append(")");
        builder.append(" ts:" + timestamp);
        return builder.toString();
    }
    @JsonIgnore
    public long getMaxVersion() {
        long max = -1;
        for(ClockEntry entry: versions)
            max = Math.max(entry.getVersion(), max);
        return max;
    }
    public VectorClock merge(VectorClock clock) {
        VectorClock newClock = new VectorClock();
        int i = 0;
        int j = 0;
        while(i < this.versions.size() && j < clock.versions.size()) {
            ClockEntry v1 = this.versions.get(i);
            ClockEntry v2 = clock.versions.get(j);
            if(v1.getNodeId() == v2.getNodeId()) {
                newClock.versions.add(new ClockEntry(v1.getNodeId(), Math.max(v1.getVersion(),
                                                                              v2.getVersion())));
                i++;
                j++;
            } else if(v1.getNodeId() < v2.getNodeId()) {
                newClock.versions.add(v1.clone());
                i++;
            } else {
                newClock.versions.add(v2.clone());
                j++;
            }
        }
        for(int k = i; k < this.versions.size(); k++)
            newClock.versions.add(this.versions.get(k).clone());
        for(int k = j; k < clock.versions.size(); k++)
            newClock.versions.add(clock.versions.get(k).clone());
        return newClock;
    }
    @Override
    public Occurred compare(IVersion v) {
        if(!(v instanceof VectorClock))
            throw new IllegalArgumentException("Cannot compare Versions of different types.");
        return compare(this, (VectorClock) v);
    }
    public static Occurred compare(VectorClock v1, VectorClock v2) {
        if(v1 == null || v2 == null)
            throw new IllegalArgumentException("Can't compare null vector clocks!");
        boolean v1Bigger = false;
        boolean v2Bigger = false;
        int p1 = 0;
        int p2 = 0;
        while(p1 < v1.versions.size() && p2 < v2.versions.size()) {
            ClockEntry ver1 = v1.versions.get(p1);
            ClockEntry ver2 = v2.versions.get(p2);
            if(ver1.getNodeId() == ver2.getNodeId()) {
                if(ver1.getVersion() > ver2.getVersion())
                    v1Bigger = true;
                else if(ver2.getVersion() > ver1.getVersion())
                    v2Bigger = true;
                p1++;
                p2++;
            } else if(ver1.getNodeId() > ver2.getNodeId()) {
                v2Bigger = true;
                p2++;
            } else {
                v1Bigger = true;
                p1++;
            }
        }
        if(p1 < v1.versions.size())
            v1Bigger = true;
        else if(p2 < v2.versions.size())
            v2Bigger = true;
        if(!v1Bigger && !v2Bigger)
            return Occurred.BEFORE;
        else if(v1Bigger && !v2Bigger)
            return Occurred.AFTER;
        else if(!v1Bigger && v2Bigger)
            return Occurred.BEFORE;
        else
            return Occurred.CONCURRENTLY;
    }
    public long getTimestamp() {
        return this.timestamp;
    }
    public List<ClockEntry> getEntries() {
        return Collections.unmodifiableList(this.versions);
    }
}
