package net.floodlightcontroller.storage;
import java.util.Iterator;
import java.util.NoSuchElementException;
public class ResultSetIterator implements Iterator<IResultSet> {
    private IResultSet resultSet;
    private boolean hasAnother;
    private boolean peekedAtNext;
    public ResultSetIterator(IResultSet resultSet) {
        this.resultSet = resultSet;
        this.peekedAtNext = false;
    }
    @Override
    public IResultSet next() {
        if (!peekedAtNext) {
            hasAnother = resultSet.next();
        }
        peekedAtNext = false;
        if (!hasAnother)
            throw new NoSuchElementException();
        return resultSet;
    }
    @Override
    public boolean hasNext() {
        if (!peekedAtNext) {
            hasAnother = resultSet.next();
            peekedAtNext = true;
        }
        return hasAnother;
    }
    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
