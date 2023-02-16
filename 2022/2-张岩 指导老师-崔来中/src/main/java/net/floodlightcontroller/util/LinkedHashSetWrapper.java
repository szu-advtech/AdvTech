package net.floodlightcontroller.util;
import java.util.Collection;
import java.util.LinkedHashSet;
import com.google.common.collect.ForwardingCollection;
public class LinkedHashSetWrapper<E>
        extends ForwardingCollection<E> implements OrderedCollection<E> {
    private final Collection<E> delegate;
    public LinkedHashSetWrapper() {
        super();
        this.delegate = new LinkedHashSet<E>();
    }
    public LinkedHashSetWrapper(Collection<? extends E> c) {
        super();
        this.delegate = new LinkedHashSet<E>(c);
    }
    @Override
    protected Collection<E> delegate() {
        return this.delegate;
    }
}
