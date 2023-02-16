import lombok.Getter;
import lombok.Setter;
import java.util.*;

/**
 * use to calculate something about allocation
 */
@Getter
@Setter
public class Solution extends HashMap<Container, LinkedList<Allocation>> {
    private final HashMap<Task, Allocation> revMapping = new HashMap<>(); //reverseMapping
    private final HashMap<Task, ArrayList<Container>> holdMapping = new HashMap<>();

    public Solution() {
        super();
        VM.resetInternalId();
    }

    //isEnd denotes whether the task is placed at the end, or the beginning
    public void addTaskToContainer(Container container, Task task, double startTime, boolean isEnd) {
        if(!this.containsKey(container))
            this.put(container, new LinkedList<>());

        Allocation alloc = new Allocation(container, task, startTime);

        //check whether there is time conflict
        boolean conflict = false;
        double startTime1 = alloc.getStartTime();
        double finishTime1 = alloc.getFinishTime();

        for(Allocation prevAlloc : this.get(container)){
            double startTime2 = prevAlloc.getStartTime();
            double finishTime2 = prevAlloc.getFinishTime();
            if ((startTime1 > startTime2 && startTime2 > finishTime1)      //startTime2 is between startTime1 and finishTime1
                    || (startTime2 > startTime1 && startTime1 > finishTime2)) { //startTime1 is between startTime2 and finishTime2
                conflict = true;
                break;
            }
        }
        if(conflict)
            throw new RuntimeException("Critical Error: Allocation conflicts");
        if(isEnd)
            this.get(container).add(alloc);
        else
            this.get(container).add(0, alloc);
        container.setUsed(true);
        revMapping.put(task, alloc);
    }

    public void updateTaskConfig(Task task) {
        //when you update the configuration, you should update the next task's EST(the chain influence)
        Allocation alloc = revMapping.get(task);
        Container container = alloc.getContainer();
        alloc.setStartTime(calEST(task, container));
        double newFinishTime = alloc.getStartTime() + alloc.getExecuteTime();
        alloc.setFinishTime(newFinishTime);
    }

    //calculate Earliest Starting Time(EST) of task on Container
    public double calEST(Task task, Container container) {
        double EST = 0;
        //max{AFT + TT}
        for(Edge inEdge : task.getInEdges()){
            Task parent = inEdge.getSource();
            Allocation alloc = revMapping.get(parent);
            Container parentContainer = alloc.getContainer();
            double arrivalTime = alloc.getFinishTime();

            if( parentContainer != container )
                arrivalTime += (double) (inEdge.getDataSize() / VM.NETWORK_SPEED); //still use VM's network
            EST = Math.max(EST, arrivalTime);
        }
        //max{avail} avail = max{IT + T, AFT}
        /*if(container == null)
            EST = Math.max(EST, VM.LAUNCH_TIME);
        else
            EST = Math.max(EST, this.getContainerReadyTime(container)); //AFT of this container*/
        return EST;
    }

    //VM's Lease Start Time(LST) and Finish Time(LFT) are calculated based on allocations
    public double getContainerLeaseStartTime(Container container) {
        if(this.get(container).size() == 0)
            return VM.LAUNCH_TIME;
        else {
            Task firstTask = this.get(container).get(0).getTask();
            double ftStartTime = this.get(container).get(0).getStartTime(); // startTime of first task

            //find the maxTransferTime because this task must wait for their data
            double maxTransferTime = 0;
            for(Edge e : firstTask.getInEdges()) {
                Allocation alloc = revMapping.get(e.getSource());
                if(alloc == null || alloc.getContainer() != container)		// parentTask's container != container
                    maxTransferTime = Math.max(maxTransferTime, (double) (e.getDataSize() / VM.NETWORK_SPEED));
            }
            return ftStartTime - maxTransferTime;
        }
    }

    public double getContainerLeaseEndTime(Container container) {
        if(this.get(container)== null || this.get(container).size() == 0)
            return VM.LAUNCH_TIME;
        else {
            LinkedList<Allocation> allocations = this.get(container);
            //get the last task
            Task lastTask = allocations.get(allocations.size()-1).getTask();
            double ltFinishTime = allocations.get(allocations.size()-1).getFinishTime(); // finishTime of last task

            double maxTransferTime = 0;
            for(Edge e : lastTask.getOutEdges()) {
                Allocation alloc = revMapping.get(e.getDestination());
                if(alloc == null || alloc.getContainer() != container)		// childTask's container != container
                    maxTransferTime = Math.max(maxTransferTime, (double) (e.getDataSize() / VM.NETWORK_SPEED));
            }
            return ltFinishTime + maxTransferTime;
        }
    }

    //note the difference between VMReadyTime and VMLeaseEndTime. It means this VM can be use for the next allocation
    public double getContainerReadyTime(Container container) {		//finish time of the last task
        if(this.get(container)== null || this.get(container).size() == 0)
            return VM.LAUNCH_TIME;
        else {
            LinkedList<Allocation> allocations = this.get(container);
            return allocations.get(allocations.size()-1).getFinishTime();
        }
    }

    public double calCost(boolean flag) {
        double totalCost = 0;
        for(Container container : this.keySet()){
            if (!container.isUsed()) continue;
            double containerCost = flag ? calContainerCostByRatio(container) :calContainerCostBySF(container);
            totalCost += containerCost;
        }
        return totalCost;
    }

    private double calContainerCostBySF(Container container) {
        return container.getVMCostSingleFit() * Math.ceil((this.getContainerLeaseEndTime(container) - this.getContainerLeaseStartTime(container)) / VM.INTERVAL);
    }


    private double calContainerCostByRatio(Container container) {
        return container.getUnitCost() * Math.ceil((this.getContainerLeaseEndTime(container) - this.getContainerLeaseStartTime(container)) / VM.INTERVAL);
    }

    //max{EFT}
    public double calMakespan() {
        double makespan = -1;
        for(Container container : this.keySet()) {
            double finishTime = -1;
            LinkedList<Allocation> allocations = this.get(container);
            for (Allocation alloc : allocations) {
                finishTime = alloc.getFinishTime(); //EST+ET
                if(finishTime > makespan) //max{EFT}
                    makespan = finishTime;
            }
        }
        return makespan;
    }
}
