package org.sdnplatform.sync.error;
public class RemoteStoreException extends SyncException {
    private static final long serialVersionUID = -8098015934951853774L;
    public RemoteStoreException() {
        super();
    }
    public RemoteStoreException(String message, Throwable cause) {
        super(message, cause);
    }
    public RemoteStoreException(String message) {
        super(message);
    }
    public RemoteStoreException(Throwable cause) {
        super(cause);
    }
}
