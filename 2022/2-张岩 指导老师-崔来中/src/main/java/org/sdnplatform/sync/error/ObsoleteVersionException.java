package org.sdnplatform.sync.error;
public class ObsoleteVersionException extends SyncException {
    private static final long serialVersionUID = 7128132048300845832L;
    public ObsoleteVersionException() {
        super();
    }
    public ObsoleteVersionException(String message, Throwable cause) {
        super(message, cause);
    }
    public ObsoleteVersionException(String message) {
        super(message);
    }
    public ObsoleteVersionException(Throwable cause) {
        super(cause);
    }
    @Override
    public ErrorType getErrorType() {
        return ErrorType.OBSOLETE_VERSION;
    }
}
