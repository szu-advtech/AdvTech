package net.floodlightcontroller.core.internal;
public class OFAuxException extends SwitchStateException{
        private static final long serialVersionUID = 8452081020837079086L;
        public OFAuxException() {
            super();
        }
        public OFAuxException(String arg0) {
            super(arg0);
        }
        public OFAuxException(Throwable arg0) {
            super(arg0);
        }
}