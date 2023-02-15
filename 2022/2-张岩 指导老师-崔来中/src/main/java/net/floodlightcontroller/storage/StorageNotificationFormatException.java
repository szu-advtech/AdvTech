package net.floodlightcontroller.storage;
public class StorageNotificationFormatException extends StorageException {
    private static final long serialVersionUID = 504758477518283156L;
    public StorageNotificationFormatException() {
        super("Invalid storage notification format");
    }
}
