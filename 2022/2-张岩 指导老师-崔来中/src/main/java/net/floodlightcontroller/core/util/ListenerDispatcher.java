package net.floodlightcontroller.core.util;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import net.floodlightcontroller.core.IListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class ListenerDispatcher<U, T extends IListener<U>> {
    protected static final Logger logger = 
            LoggerFactory.getLogger(ListenerDispatcher.class);
    volatile List<T> listeners = new ArrayList<T>();
    private void visit(List<T> newlisteners, U type, HashSet<T> visited,
                       List<T> ordering, T listener) {
        if (!visited.contains(listener)) {
            visited.add(listener);
            for (T i : newlisteners) {
                if (ispre(type, i, listener)) {
                    visit(newlisteners, type, visited, ordering, i);
                }
            }
            ordering.add(listener);
        }
    }
    private boolean ispre(U type, T l1, T l2) {
        return (l2.isCallbackOrderingPrereq(type, l1.getName()) ||
                l1.isCallbackOrderingPostreq(type, l2.getName()));
    }
    public void addListener(U type, T listener) {
        List<T> newlisteners = new ArrayList<T>();
        if (listeners != null)
            newlisteners.addAll(listeners);
        newlisteners.add(listener);
        List<T> terminals = new ArrayList<T>();
        for (T i : newlisteners) {
            boolean isterm = true;
            for (T j : newlisteners) {
                if (ispre(type, i, j)) {
                    isterm = false;
                    break;
                }
            }
            if (isterm) {
                terminals.add(i);
            }
        }
        if (terminals.size() == 0) {
            logger.error("No listener dependency solution: " +
                         "No listeners without incoming dependencies");
            listeners = newlisteners;
            return;
        }
        HashSet<T> visited = new HashSet<T>();
        List<T> ordering = new ArrayList<T>();
        for (T term : terminals) {
            visit(newlisteners, type, visited, ordering, term);
        }
        listeners = ordering;
    }
    public void removeListener(T listener) {
        if (listeners != null) {
            List<T> newlisteners = new ArrayList<T>();
            newlisteners.addAll(listeners);
            newlisteners.remove(listener);
            listeners = newlisteners;
        }
    }
    public void clearListeners() {
        listeners = new ArrayList<T>();
    }
    public List<T> getOrderedListeners() {
        return listeners;
    }
}
