package org.sdnplatform.sync.internal.version;
import java.util.ArrayList;
import java.util.List;
import org.sdnplatform.sync.IInconsistencyResolver;
public class ChainedResolver<T> implements IInconsistencyResolver<T> {
    private List<IInconsistencyResolver<T>> resolvers;
    @SafeVarargs
	public ChainedResolver(IInconsistencyResolver<T>... resolvers) {
        this.resolvers = new ArrayList<IInconsistencyResolver<T>>(resolvers.length);
        for(IInconsistencyResolver<T> resolver: resolvers)
            this.resolvers.add(resolver);
    }
    public List<T> resolveConflicts(List<T> items) {
        for(IInconsistencyResolver<T> resolver: resolvers) {
            if(items.size() <= 1)
                return items;
            else
                items = resolver.resolveConflicts(items);
        }
        return items;
    }
    @Override
    public boolean equals(Object o) {
        if(this == o)
            return true;
        if(o == null || getClass() != o.getClass())
            return false;
        ChainedResolver<?> that = (ChainedResolver<?>) o;
        if(resolvers != null
                ? !resolvers.equals(that.resolvers)
                : that.resolvers != null)
            return false;
        return true;
    }
    @Override
    public int hashCode() {
        return resolvers != null ? resolvers.hashCode() : 0;
    }
}
