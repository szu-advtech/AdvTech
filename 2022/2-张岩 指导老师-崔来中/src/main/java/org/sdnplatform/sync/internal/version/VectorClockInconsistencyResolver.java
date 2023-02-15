package org.sdnplatform.sync.internal.version;
import java.util.List;
import java.util.ListIterator;
import org.sdnplatform.sync.IInconsistencyResolver;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.IVersion.Occurred;
import com.google.common.collect.Lists;
public class VectorClockInconsistencyResolver<T>
    implements IInconsistencyResolver<Versioned<T>> {
    public List<Versioned<T>> resolveConflicts(List<Versioned<T>> items) {
        int size = items.size();
        if(size <= 1)
            return items;
        List<Versioned<T>> newItems = Lists.newArrayList();
        for(Versioned<T> v1: items) {
            boolean found = false;
            for(ListIterator<Versioned<T>> it2 =
                    newItems.listIterator(); it2.hasNext();) {
                Versioned<T> v2 = it2.next();
                Occurred compare = v1.getVersion().compare(v2.getVersion());
                if(compare == Occurred.AFTER) {
                    if(found)
                        it2.remove();
                    else
                        it2.set(v1);
                }
                if(compare != Occurred.CONCURRENTLY)
                    found = true;
            }
            if(!found)
                newItems.add(v1);
        }
        return newItems;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return (o != null && getClass() == o.getClass());
    }
    @Override
    public int hashCode() {
        return getClass().hashCode();
    }
}
