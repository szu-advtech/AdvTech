package org.sdnplatform.sync.internal.version;
import java.util.Collections;
import java.util.List;
import org.sdnplatform.sync.IInconsistencyResolver;
import org.sdnplatform.sync.Versioned;
public class TimeBasedInconsistencyResolver<T>
    implements IInconsistencyResolver<Versioned<T>> {
    public List<Versioned<T>> resolveConflicts(List<Versioned<T>> items) {
        if(items.size() <= 1) {
            return items;
        } else {
            Versioned<T> max = items.get(0);
            long maxTime =
                    ((VectorClock) items.get(0).getVersion()).getTimestamp();
            VectorClock maxClock = ((VectorClock) items.get(0).getVersion());
            for(Versioned<T> versioned: items) {
                VectorClock clock = (VectorClock) versioned.getVersion();
                if(clock.getTimestamp() > maxTime) {
                    max = versioned;
                    maxTime = ((VectorClock) versioned.getVersion()).
                            getTimestamp();
                }
                maxClock = maxClock.merge(clock);
            }
            Versioned<T> maxTimeClockVersioned =
                    new Versioned<T>(max.getValue(), maxClock);
            return Collections.singletonList(maxTimeClockVersioned);
        }
    }
    @Override
    public boolean equals(Object o) {
        if(this == o)
            return true;
        return (o != null && getClass() == o.getClass());
    }
    @Override
    public int hashCode() {
        return getClass().hashCode();
    }
}
