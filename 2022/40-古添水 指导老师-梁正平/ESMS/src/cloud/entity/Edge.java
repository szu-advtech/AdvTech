import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.util.Comparator;
import java.util.List;

/**
 * use to represent the association between tasks
 */
@Getter
@Setter
public class Edge {
    private Task source;
    private Task destination;
    private long dataSize;

    public Edge(Task source, Task destination) {
        this.source = source;
        this.destination = destination;
    }

    @Override
    public String toString() {
        return "Edge [source=" + source + ", destination=" + destination + ", size=" + dataSize + "]";
    }

    @AllArgsConstructor
    static class EdgeComparator implements Comparator<Edge> {
        boolean isDestination;
        List<Task> topoList;

        /*
        order the task(whatever it is dest or source) by the index of topoSort-List
        if a set have many edges,there will be the small index first
         */
        @Override
        public int compare(Edge o1, Edge o2) {
            Task task1 = isDestination ? o1.getDestination() : o1.getSource();
            Task task2 = isDestination ? o2.getDestination() : o2.getSource();
            int index1 = topoList.indexOf(task1);
            int index2 = topoList.indexOf(task2);
            if(index1 > index2)
                return 1;
            else if(index1 < index2)
                return -1;
            return 0;
        }
    }
}
