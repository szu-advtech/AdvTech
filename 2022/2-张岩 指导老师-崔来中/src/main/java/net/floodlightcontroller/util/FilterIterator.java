package net.floodlightcontroller.util;
import java.util.Iterator;
import java.util.NoSuchElementException;
public abstract class FilterIterator<T> implements Iterator<T> {
    protected Iterator<T> subIterator;
    protected T next;
    public FilterIterator(Iterator<T> subIterator) {
        super();
        this.subIterator = subIterator;
    }
    protected abstract boolean matches(T value);
    @Override
    public boolean hasNext() {
        if (next != null) return true;
        while (subIterator.hasNext()) {
            next = subIterator.next();
            if (matches(next))
                return true;
        }
        next = null;
        return false;
    }
    @Override
    public T next() {
        if (hasNext()) {
            T cur = next;
            next = null;
            return cur;
        }
        throw new NoSuchElementException();
    }
    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
