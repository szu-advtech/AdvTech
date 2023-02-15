import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
public class Allocation {
    private Task task;
    private Container container;
    private double startTime;
    private double executeTime;
    private double finishTime;
    private double laxity;

    public Allocation(Container container, Task task, double startTime) {
        this.container = container;
        this.task = task;
        this.startTime = startTime;
        if(container != null && task !=null)
            this.finishTime = startTime + (task.getTaskSize() / container.getECU());
    }

    public double getExecuteTime() {
        return task.getTaskSize() / container.getECU();
    }
}
