package org.sdnplatform.sync;
import java.io.Closeable;
import java.util.Iterator;
public interface IClosableIterator<T> extends Iterator<T>,Closeable {
    public void close();
}
