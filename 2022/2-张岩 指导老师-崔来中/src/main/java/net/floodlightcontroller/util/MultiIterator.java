package net.floodlightcontroller.util;
import java.util.Iterator;
import java.util.NoSuchElementException;
public class MultiIterator<T> implements Iterator<T> {
    Iterator<Iterator<T>> subIterator;
    Iterator<T> current = null;
    public MultiIterator(Iterator<Iterator<T>> subIterator) {
        super();
        this.subIterator = subIterator;
    }
    @Override
    public boolean hasNext() {
        if (current == null) {
            if (subIterator.hasNext()) {
                current = subIterator.next();
            } else {
                return false;
            }
        }
        while (!current.hasNext() && subIterator.hasNext()) {
            current = subIterator.next();
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
