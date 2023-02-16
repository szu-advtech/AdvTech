package net.floodlightcontroller.core;
public interface Deliverable<T> {
    public static enum Status {
        DONE,
        CONTINUE
    }
    public void deliver(T msg);
    void deliverError(Throwable cause);
    boolean isDone();
    boolean cancel(boolean mayInterruptIfRunning);
}
