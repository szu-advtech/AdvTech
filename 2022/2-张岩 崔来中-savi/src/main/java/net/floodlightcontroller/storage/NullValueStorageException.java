package net.floodlightcontroller.storage;
public class NullValueStorageException extends StorageException {
    private static final long serialVersionUID = 897572085681189926L;
    private static String makeExceptionMessage(String columnName) {
        String message = "Null column value could not be converted to built-in type";
        if (columnName != null) {
            message += ": column name = ";
            message += columnName;
        }
        return message;
    }
    public NullValueStorageException() {
        super(makeExceptionMessage(null));
    }
    public NullValueStorageException(String columnName) {
        super(makeExceptionMessage(columnName));
    }
    public NullValueStorageException(String columnName, Throwable exc) {
        super(makeExceptionMessage(columnName), exc);
    }
}
