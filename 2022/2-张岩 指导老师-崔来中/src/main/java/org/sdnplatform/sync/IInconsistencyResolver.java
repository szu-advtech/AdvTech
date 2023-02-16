package org.sdnplatform.sync;
import java.util.List;
public interface IInconsistencyResolver<T> {
    public List<T> resolveConflicts(List<T> items);
}
