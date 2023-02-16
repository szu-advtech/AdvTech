package net.floodlightcontroller.util;
import java.util.Iterator;
import java.util.NoSuchElementException;
public class IterableIterator<T> implements Iterator<T> {
    Iterator<? extends Iterable<T>> subIterator;
    Iterator<T> current = null;
    public IterableIterator(Iterator<? extends Iterable<T>> subIterator) {
        super();
        this.subIterator = subIterator;
    }
    @Override
    public boolean hasNext() {
        if (current == null) {
            if (subIterator.hasNext()) {
                current = subIterator.next().iterator();
            } else {
                return false;
            }
        }
        while (!current.hasNext() && subIterator.hasNext()) {
            current = subIterator.next().iterator();
        }
        return current.hasNext();
    }
    @Override
    public T next() {
        if (hasNext())
            return current.next();
        throw new NoSuchElementException();
    }
    @Override
    public void remove() {
        if (hasNext())
            current.remove();
        throw new NoSuchElementException();
    }
}
