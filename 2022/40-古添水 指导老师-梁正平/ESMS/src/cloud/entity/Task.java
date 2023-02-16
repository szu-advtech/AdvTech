import lombok.Getter;
import lombok.Setter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Getter
@Setter
public class Task {
    private static int internalId = 0;

    private int id;
    private String name;
    private double taskSize;
    private double rank;
    private double urgency;
    private double subDDL;

    private int topoCount;
    private double bLevel; 	//blevel
    private double sLevel;
    private double ve; //earlist happened
    private double vl;
    private int hop;

    private List<Edge> outEdges = new ArrayList<>();
    private List<Edge> inEdges = new ArrayList<>();

    public static void resetInternalId() { internalId = 0; }

    public Task(String name, double taskSize) {
        this.id = internalId++;
        this.name = name;
        this.taskSize = taskSize;
    }

    public void insertInEdge(Edge e){
        if(e.getDestination()!=this)
            throw new RuntimeException();
        inEdges.add(e);
    }
    public void insertOutEdge(Edge e){
        if(e.getSource()!=this)
            throw new RuntimeException();
        outEdges.add(e);
    }

    static class BLevelComparator implements Comparator<Task> {
        public int compare(Task o1, Task o2) {
            // to keep entry node ranking last, and exit node first
            if(o1.getName().equals("entry") || o2.getName().equals("exit"))
                return 1;
            if(o1.getName().equals("exit") || o2.getName().equals("entry"))
                return -1;
            return Double.compare(o1.getBLevel(), o2.getBLevel());
        }
    }

    // used to calculate the largest number of parallel tasks in workflow
    static class ParallelComparator implements Comparator<Task>{
        public int compare(Task o1, Task o2) {
            int d1 = o1.getOutEdges().size() - o1.getInEdges().size();
            int d2 = o2.getOutEdges().size() - o2.getInEdges().size();
            // because of the use of PriorityQueue, here the comparison is reverse
            return Integer.compare(d2, d1);
        }
    }

    @Override
    public String toString() {
        return "Task [id=" + name + ", taskSize=" + taskSize + ", subDDL="+ subDDL + ", urgency=" + urgency +"]";
    }
}
