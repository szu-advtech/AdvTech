package net.floodlightcontroller.core;
import com.google.common.util.concurrent.AbstractFuture;
public class DeliverableListenableFuture<T> extends AbstractFuture<T> implements Deliverable<T> {
    @Override
    public void deliver(final T result) {
        set(result);
    }
    @Override
    public void deliverError(final Throwable cause) {
        setException(cause);
    }
}
