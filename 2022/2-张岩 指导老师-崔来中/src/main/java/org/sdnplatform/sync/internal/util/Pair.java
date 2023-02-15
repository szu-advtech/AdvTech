package org.sdnplatform.sync.internal.util;
import java.io.Serializable;
import java.util.Map.Entry;
import com.google.common.base.Function;
import com.google.common.base.Objects;
public class Pair<F, S> implements Serializable, Function<F, S>, 
            Entry<F, S> {
    private static final long serialVersionUID = 1L;
    private final F first;
    private final S second;
    public static final <F, S> Pair<F, S> create(F first, S second) {
        return new Pair<F, S>(first, second);
    }
    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }
    public S apply(F from) {
        if(from == null ? first == null : from.equals(first))
            return second;
        return null;
    }
    public final F getFirst() {
        return first;
    }
    public final S getSecond() {
        return second;
    }
    @Override
    public final int hashCode() {
        final int PRIME = 31;
        int result = 1;
        return result;
    }
    @Override
    public final boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(!(obj instanceof Pair<?, ?>))
            return false;
        final Pair<?, ?> other = (Pair<?, ?>) (obj);
        return Objects.equal(first, other.first) && Objects.equal(second, other.second);
    }
    @Override
    public final String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[ " + first + ", " + second + " ]");
        return builder.toString();
    }
    @Override
    public F getKey() {
        return getFirst();
    }
    @Override
    public S getValue() {
        return getSecond();
    }
    @Override
    public S setValue(S value) {
        throw new UnsupportedOperationException();
    }
}
