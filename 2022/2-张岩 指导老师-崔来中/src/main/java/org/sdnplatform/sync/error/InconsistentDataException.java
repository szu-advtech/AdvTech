package org.sdnplatform.sync.error;
import java.util.List;
public class InconsistentDataException extends SyncException {
    private static final long serialVersionUID = 1050277622160468516L;
    List<?> unresolvedVersions;
    public InconsistentDataException(String message, List<?> versions) {
        super(message);
        this.unresolvedVersions = versions;
    }
    public List<?> getUnresolvedVersions() {
        return unresolvedVersions;
    }
    @Override
    public ErrorType getErrorType() {
        return ErrorType.INCONSISTENT_DATA;
    }
}
