package org.sdnplatform.sync.error;
public class SyncRuntimeException extends RuntimeException {
    private static final long serialVersionUID = -5357245946596447913L;
    public SyncRuntimeException(String message, SyncException cause) {
        super(message, cause);
    }
    public SyncRuntimeException(SyncException cause) {
        super(cause);
    }
}
