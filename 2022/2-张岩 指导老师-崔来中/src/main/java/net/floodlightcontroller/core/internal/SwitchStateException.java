package net.floodlightcontroller.core.internal;
public class SwitchStateException extends IllegalArgumentException {
    private static final long serialVersionUID = 9153954512470002631L;
    public SwitchStateException() {
        super();
    }
    public SwitchStateException(String arg0) {
        super(arg0);
    }
    public SwitchStateException(Throwable arg0) {
        super(arg0);
    }
}
