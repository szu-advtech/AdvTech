package org.sdnplatform.sync;
public interface IVersion {
    public enum Occurred {
        BEFORE,
        AFTER,
        CONCURRENTLY
    }
    public Occurred compare(IVersion v);
}
