package net.floodlightcontroller.storage;
public class TypeMismatchStorageException extends StorageException {
    private static final long serialVersionUID = -7923586656854871345L;
    private static String makeExceptionMessage(String requestedType, String actualType, String columnName) {
        if (requestedType == null)
            requestedType = "???";
        if (actualType == null)
            actualType = "???";
        if (columnName == null)
            columnName = "???";
        String message = "The requested type (" + requestedType + ") does not match the actual type (" + actualType + ") of the value for column \"" + columnName + "\".";
        return message;
    }
    public TypeMismatchStorageException() {
        super(makeExceptionMessage(null, null, null));
    }
    public TypeMismatchStorageException(String requestedType, String actualType, String columnName) {
        super(makeExceptionMessage(requestedType, actualType, columnName));
    }
}
