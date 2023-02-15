package net.floodlightcontroller.debugcounter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.TreeMap;
import java.util.regex.Pattern;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
class CounterNode implements Iterable<DebugCounterImpl> {
    private static final String QUOTED_SEP = Pattern.quote("/");
    private static final Logger log = LoggerFactory.getLogger(CounterNode.class);
    private final String hierarchy;
    private final List<String> hierarchyElements;
    private final DebugCounterImpl counter;
    private final TreeMap<String, CounterNode> children = new TreeMap<>();
    static List<String> getHierarchyElements(String moduleName, String counterHierarchy) {
        DebugCounterServiceImpl.verifyModuleNameSanity(moduleName);
        List<String> ret = new ArrayList<>();
        ret.add(moduleName);
        if (counterHierarchy == null || counterHierarchy.isEmpty()) {
            return ret;
        }
        for (String element : counterHierarchy.split(QUOTED_SEP)) {
            ret.add(element);
        }
        return ret;
    }
    private CounterNode(List<String> hierarchyElements, DebugCounterImpl counter) {
        super();
        this.hierarchyElements = ImmutableList.copyOf(hierarchyElements);
        this.hierarchy = Joiner.on("/").join(hierarchyElements);
        this.counter = counter;
    }
    public static CounterNode newTree() {
        return new CounterNode(ImmutableList.<String>of(), null);
    }
    private void verifyIsRoot() {
        if (hierarchyElements.size() != 0) {
            throw new IllegalStateException("This is not the root. Can "
                    + "only call addCounter() on the root node. Current node: "
                    + hierarchy);
        }
    }
    @Nonnull
    String getHierarchy() {
        return hierarchy;
    }
    @Nonnull
    List<String> getHierarchyElements() {
        return hierarchyElements;
    }
    @Nullable
    DebugCounterImpl getCounter() {
        return counter;
    }
    void resetHierarchy() {
        for (DebugCounterImpl cur: this) {
            cur.reset();
        }
    }
    Iterable<DebugCounterImpl> getCountersInHierarchy() {
        return this;
    }
    CounterNode lookup(List<String> hierarchyElements) {
        CounterNode cur = this;
        for (String element: hierarchyElements) {
            cur = cur.children.get(element);
            if (cur == null) {
                break;
            }
        }
        return cur;
    }
    CounterNode remove(List<String> hierarchyElements) {
        CounterNode cur = this;
        if (hierarchyElements.isEmpty()) {
        	log.error("Cannot remove a CounterNode from an empty list of hierarchy elements. Returning null.");
        	return null;
        } 
        String keyToRemove = hierarchyElements.remove(hierarchyElements.size() - 1); 
        for (String element: hierarchyElements) {
            cur = cur.children.get(element);
            if (cur == null) {
                break;
            }
        }
        CounterNode removed = null;
        if (cur != null) {
        	removed = cur.children.remove(keyToRemove);
        }
        return removed;
    }
    boolean addModule(@Nonnull String moduleName) {
        verifyIsRoot();
        if (children.containsKey(moduleName)) {
            children.get(moduleName).resetHierarchy();
            return false;
        } else {
            CounterNode newNode =
                    new CounterNode(ImmutableList.of(moduleName), null);
            children.put(moduleName, newNode);
            return true;
        }
    }
    @Nullable
    DebugCounterImpl addCounter(@Nonnull DebugCounterImpl counter) {
        verifyIsRoot();
        ArrayList<String> path = new ArrayList<>();
        path.add(counter.getModuleName());
        for (String element: counter.getCounterHierarchy().split(QUOTED_SEP)) {
            path.add(element);
        }
        String newCounterName = path.get(path.size()-1);
        CounterNode parent = lookup(path.subList(0, path.size()-1));
        if (parent == null) {
            throw new IllegalArgumentException("Missing hierarchy level for "
                    + "counter: " + counter.getModuleName() + " "
                    + counter.getCounterHierarchy());
        }
        if (parent.children.containsKey(newCounterName)) {
            CounterNode old = parent.children.get(newCounterName);
            old.resetHierarchy();
            return old.counter;
        } else {
            CounterNode newNode = new CounterNode(path, counter);
            parent.children.put(newCounterName, newNode);
            return null;
        }
    }
    private final static class CounterIterator implements Iterator<DebugCounterImpl> {
        ArrayDeque<CounterNode> stack = new ArrayDeque<>();
        CounterNode curNode = null;
        private CounterIterator(CounterNode root) {
            stack.push(root);
            gotoNextNode();
        }
        private void gotoNextNode() {
            while (true) {
                curNode = null;
                if (stack.isEmpty()) {
                    break;
                }
                curNode = stack.pop();
                for (CounterNode child: curNode.children.descendingMap().values()) {
                    stack.push(child);
                }
                if (curNode.counter != null) {
                    break;
                }
            }
        }
        @Override
        public boolean hasNext() {
            return curNode != null;
        }
        @Override
        public DebugCounterImpl next() {
            if (curNode == null) {
                throw new NoSuchElementException();
            }
            DebugCounterImpl ret = curNode.counter;
            gotoNextNode();
            return ret;
        }
        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }
    @Override
    public Iterator<DebugCounterImpl> iterator() {
        return new CounterIterator(this);
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
                 + ((children == null) ? 0 : children.hashCode());
                 + ((counter == null) ? 0 : counter.hashCode());
                 + ((hierarchy == null) ? 0 : hierarchy.hashCode());
        result = prime
                 + ((hierarchyElements == null) ? 0
                                               : hierarchyElements.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        CounterNode other = (CounterNode) obj;
        if (children == null) {
            if (other.children != null) return false;
        } else if (!children.equals(other.children)) return false;
        if (counter == null) {
            if (other.counter != null) return false;
        } else if (!counter.equals(other.counter)) return false;
        if (hierarchy == null) {
            if (other.hierarchy != null) return false;
        } else if (!hierarchy.equals(other.hierarchy)) return false;
        if (hierarchyElements == null) {
            if (other.hierarchyElements != null) return false;
        } else if (!hierarchyElements.equals(other.hierarchyElements))
                                                                      return false;
        return true;
    }
    @Override
    public String toString() {
        return toString(0);
    }
    public String toString(int indent) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < indent; i++) {
            builder.append(" ");
        }
        builder.append("hierarchy=");
        builder.append(hierarchy);
        builder.append(", counter=");
        builder.append(counter);
        builder.append(", children=");
        builder.append(children.keySet());
        builder.append("\n");
        for (CounterNode child: children.values()) {
            builder.append(child.toString(indent + 3));
        }
        return builder.toString();
    }
}
