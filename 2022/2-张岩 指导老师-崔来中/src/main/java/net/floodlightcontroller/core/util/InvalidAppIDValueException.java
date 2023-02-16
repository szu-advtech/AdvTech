package net.floodlightcontroller.core.util;
public class InvalidAppIDValueException extends AppIDException {
    private static final long serialVersionUID = -1866481021012360918L;
    public InvalidAppIDValueException(long invalidId) {
        super("Application ID " + invalidId + "is not valid");
    }
}
