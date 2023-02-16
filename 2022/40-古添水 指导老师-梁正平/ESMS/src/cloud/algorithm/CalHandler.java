import java.util.*;

public class CalHandler {
    private static Workflow workflow;
    private static Solution solution;
    private static List<VM> vms;
    private static HashMap<Task, Allocation> revMapping;
    private static HashMap<Task, ArrayList<Container>> holdMapping = new HashMap<>();

    public static void setCondition(Workflow workflow, Solution solution, List<VM> vms) {
        CalHandler.workflow = workflow;
        CalHandler.solution = solution;
        CalHandler.vms = vms;
        revMapping = solution.getRevMapping();
        holdMapping = solution.getHoldMapping();
    }

    private static void firstAllocateTask() {
        int wfLen = workflow.size();
        for (int j = 0; j < wfLen; j++) {
            Task task = workflow.get(j);

            Random random = new Random();
            int nums = random.nextInt(5)+1;
            ArrayList<Container> existList = holdMapping.get(task);

            //create many microservice
            for (int k = 0; k < nums; k++) {
                Container container = new Container(VM.SLOWEST);
                if (existList == null) {
                    existList = new ArrayList<>();
                    holdMapping.put(task, existList);
                }
                existList.add(container);
            }

            //choose one to take the task
            Container container = existList.get(random.nextInt(nums));
            container.setUsed(true);
            double startTime = solution.calEST(task, container);
            solution.addTaskToContainer(container, task, startTime, j == wfLen - 1);

            //whether the existed vm load the image
            if (random.nextInt(6) < 3) vms.get(random.nextInt(VM.TYPE_NO)).getImages().add(container);
        }
    }

    private static void updateTaskConfig(Task task) {
        int index = workflow.indexOf(task);
        int size = workflow.size();
        for (int j = index; j < size; j++) {
            Task task4update = workflow.get(j);
            solution.updateTaskConfig(task4update);
        }
    }

    public static void calCE() {
        firstAllocateTask();

        double deadline = workflow.getDeadline();
        double cost = solution.calCost(true);
        double makespan = solution.calMakespan();

        System.out.println("Workflow's deadline: " + deadline);
        System.out.println("first makespan based on first configuration: " + makespan);

        while (makespan > deadline) {
            double gain = 0.0; //current biggest gain
            Container c4update = null; //the container should be updated by increasing
            Task t4update = null;
            for (Task task : workflow) {
                Allocation alloc = revMapping.get(task);
                Container container = alloc.getContainer();

                container.increaseECU();
                updateTaskConfig(task);

                double gainj = 0.0;
                double costj = solution.calCost(true);
                double makespanj = solution.calMakespan();

                if (Math.abs(cost - costj) < 1e-6) {
                    gainj = makespan - makespanj;
                } else if (costj < cost) {
                    gainj = cost - costj;
                } else {
                    gainj = (makespan - makespanj) / (costj - cost);
                }
                if (gainj > gain) { //select the biggest gain
                    c4update = container;
                    t4update = task;
                    gain = gainj;
                }

                container.decreaseECU();
                updateTaskConfig(task);
            }
            //It finds a CEj to update the container(the gainj is the biggest)
            if (c4update != null) {
                c4update.increaseECU();
                updateTaskConfig(t4update);
                if (c4update.getECU() > VM.MAX_SPEED || Math.abs(c4update.getECU() - VM.MAX_SPEED) < 1e-6)
                    VM.MAX_SPEED = c4update.getECU();
            }
            cost = solution.calCost(true);
            makespan = solution.calMakespan();

            /*System.out.println("cost:" + cost);
            System.out.println("makespan:" + makespan);*/
        }
    }

    public static void calUrgency() {
        for (Task task : workflow) {
            double sd = task.getSubDDL();
            double XFT = revMapping.get(task).getFinishTime();
            int hop = workflow.calHop(task, revMapping);
            task.setUrgency((sd - XFT) / hop);
        }
    }

    public static List<Task> getSortedTaskList() {
        List<Task> list = Arrays.asList(new Task[workflow.size()]);
        Collections.copy(list, workflow);
        list.sort((o1, o2) -> (int) (o1.getUrgency() - o2.getUrgency()));
        return list;
    }

    public static void changeTaskContainer(Task task, Container container) {
        Allocation allocation = revMapping.get(task);
        Container origin = revMapping.get(task).getContainer();
        if (origin != container) {
            origin.setUsed(false);
            container.setUsed(true);
            allocation.setContainer(container);
            updateTaskConfig(task);
        }
    }

    //invoke the container replace, should be set the container to used
    public static double calLaxity(Task task, Container container) {
        changeTaskContainer(task, container);
        Allocation allocation = revMapping.get(task);
        double EFT = allocation.getFinishTime();
        return task.getSubDDL() - EFT;
    }

    public static double calMinSpeed(Task task) {
        return task.getTaskSize() / (task.getSubDDL() - VM.IMAGE_INIT_TIME);
    }

    public static double calTotalCost(Set<VM> vmSet) {
        double cost = 0.0;
        for (VM vm : vmSet) cost += vm.getUnitCost();
        return cost;
    }

    public static Container getMinEFTInstance(Task task) {
        ArrayList<Container> containers = holdMapping.get(task);
        double maxSpeed = Double.MIN_VALUE;
        Container chosen = null;
        for (Container container : containers) {
            double ecu = container.getECU();
            if (ecu > maxSpeed) {
                maxSpeed = ecu;
                chosen = container;
            }
        }
        return chosen;
    }

    public static Container setMinEFTContainer2Task(Task task, Container c1, Container c2) {
        Container chosen = c1.getECU() > c2.getECU() ? c1 : c2;
        changeTaskContainer(task, chosen);
        return chosen;
    }

    public static List<Container> getLatestVMsContainer(HashMap<VM, List<Container>> fitMapping) {
        List<Container> map = null;
        for (Map.Entry<VM, List<Container>> entry : fitMapping.entrySet()) map = entry.getValue();
        return map;
    }
}
