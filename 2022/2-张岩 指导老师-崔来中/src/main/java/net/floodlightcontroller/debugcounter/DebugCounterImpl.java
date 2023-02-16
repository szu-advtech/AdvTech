package net.floodlightcontroller.debugcounter;
import java.util.Collection;
import java.util.Date;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nonnull;
import net.floodlightcontroller.debugcounter.IDebugCounterService.MetaData;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
class DebugCounterImpl implements IDebugCounter {
    private final String moduleName;
    private final String counterHierarchy;
    private final String description;
    private final ImmutableSet<IDebugCounterService.MetaData> metaData;
    private final AtomicLong value = new AtomicLong();
    DebugCounterImpl(@Nonnull String moduleName,
                     @Nonnull String counterHierarchy,
                     @Nonnull String description,
                     @Nonnull Collection<MetaData> metaData) {
        this.moduleName = moduleName;
        this.counterHierarchy = counterHierarchy;
        this.description = description;
        this.metaData = Sets.immutableEnumSet(metaData);
        this.lastModified.setTime(System.currentTimeMillis());
    }
    @Nonnull
    String getModuleName() {
        return moduleName;
    }
    @Nonnull
    String getCounterHierarchy() {
        return counterHierarchy;
    }
    @Nonnull
    String getDescription() {
        return description;
    }
    @Nonnull
    ImmutableSet<IDebugCounterService.MetaData> getMetaData() {
        return metaData;
    }
    @Override
    public void reset() {
        value.set(0);
        lastModified.setTime(System.currentTimeMillis());
    }
    @Override
    public void increment() {
        value.incrementAndGet();
        lastModified.setTime(System.currentTimeMillis());
    }
    @Override
    public void add(long increment) {
        if (increment < 0) {
            throw new IllegalArgumentException("increment must be > 0. Was "
                    + increment);
        }
        value.addAndGet(increment);
        lastModified.setTime(System.currentTimeMillis());
    }
    @Override
    public long getCounterValue() {
        return value.get();
    }
	@Override
	public long getLastModified() {
		return lastModified.getTime();
	}
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime
                 + ((counterHierarchy == null) ? 0
                                              : counterHierarchy.hashCode());
                 + ((description == null) ? 0 : description.hashCode());
                 + ((metaData == null) ? 0 : metaData.hashCode());
                 + ((moduleName == null) ? 0 : moduleName.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        DebugCounterImpl other = (DebugCounterImpl) obj;
        if (counterHierarchy == null) {
            if (other.counterHierarchy != null) return false;
        } else if (!counterHierarchy.equals(other.counterHierarchy))
                                                                    return false;
        if (description == null) {
            if (other.description != null) return false;
        } else if (!description.equals(other.description)) return false;
        if (metaData == null) {
            if (other.metaData != null) return false;
        } else if (!metaData.equals(other.metaData)) return false;
        if (moduleName == null) {
            if (other.moduleName != null) return false;
        } else if (!moduleName.equals(other.moduleName)) return false;
        if (value == null) {
            if (other.value != null) return false;
        } else if (value.get() != other.value.get()) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append(moduleName);
        builder.append(" ");
        builder.append(counterHierarchy);
        builder.append(", description=");
        builder.append(description);
        builder.append(", metaData=");
        builder.append(metaData);
        builder.append(", value=");
        builder.append(value);
        builder.append("]");
        return builder.toString();
    }
}
