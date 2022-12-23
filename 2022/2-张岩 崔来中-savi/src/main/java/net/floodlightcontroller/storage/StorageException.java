package net.floodlightcontroller.storage;
public class StorageException extends RuntimeException {
    static final long serialVersionUID = 7839989010156155681L;
    static private String makeExceptionMessage(String s) {
        String message = "Storage Exception";
        if (s != null) {
            message += ": ";
            message += s;
        }
        return message;
    }
    public StorageException() {
        super(makeExceptionMessage(null));
    }
    public StorageException(String s) {
        super(makeExceptionMessage(s));
    }
    public StorageException(String s, Throwable exc) {
        super(makeExceptionMessage(s), exc);
    }
}
