import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.util.*;

/**
 * Urgency-Based WorkFlow Scheduling
 */
@Getter
@Setter
@AllArgsConstructor
public class UWS {
    private Workflow workflow; //inited
    private Solution solution;
    private HashMap<Task, Allocation> revMapping;
    private HashMap<Task, ArrayList<Container>> holdMapping;
    private List<VM> vms;
    private Configuration conf;

    public List<Container> execute() {
        CalHandler.setCondition(workflow, solution, vms);
        CalHandler.calCE();

        System.out.println("---------------------------------------------------------------");
        System.out.println("After CE, it's makespan is: " + solution.calMakespan());
        System.out.println("After CE, it's cost(by single fit) is: " + solution.calCost(false));
        System.out.println("After CE, it's cost(by ratio, do not consider VM's basic resource) is: " + solution.calCost(true));
        System.out.println("---------------------------------------------------------------");

        //cal rank and subDDL
        workflow.calTaskRankBasedCE(revMapping);
        workflow.calSubDDL();
        CalHandler.calUrgency();

        List<Task> unassigned = CalHandler.getSortedTaskList();// ordered by urgency
        List<Container> newIns = new ArrayList<>();
        for (Task task : unassigned) {
            Container chosen = null; //the chosen one to the task

            Container origin = revMapping.get(task).getContainer();
            double incrCost = Double.MAX_VALUE;
            boolean laxityNeg = false;

            for (Container container : holdMapping.get(task)) {
                double laxity = CalHandler.calLaxity(task, container);
                if (laxity < 0) laxityNeg = true;
                else {
                    double currentCost = solution.calCost(true);
                    if (incrCost > currentCost) { //minimum incrCost
                        incrCost = currentCost;
                        chosen = container;
                    }
                }
            }
            if (!laxityNeg) {
                CalHandler.changeTaskContainer(task, chosen);
            } else {
                double minSpeed = CalHandler.calMinSpeed(task);
                if (minSpeed < 0 || minSpeed > VM.MAX_SPEED) {
                    Container c1 = CalHandler.getMinEFTInstance(task);
                    Container c2 = new Container(VM.MAX_SPEED);
                    holdMapping.get(task).add(c2);
                    chosen = CalHandler.setMinEFTContainer2Task(task, c1, c2);
                } else if (minSpeed < origin.getECU()) {
                    chosen = origin;
                    CalHandler.changeTaskContainer(task, origin);
                } else {
                    Container c1 = new Container(minSpeed);
                    chosen = c1;
                    CalHandler.changeTaskContainer(task, c1);
                }
            }
            assert chosen != null;
            VM vm = Fitter.bestFit(vms, chosen, false);
            if (vm == null) newIns.add(chosen);
            else conf.addVM(vm);
        }
        return newIns;
    }
}
