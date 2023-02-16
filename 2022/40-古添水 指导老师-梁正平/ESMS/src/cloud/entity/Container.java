import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;

/**
 * It can be used to represent the configuration
 */
@Getter
@Setter
public class Container implements Serializable {
    private static final long serialVersionUID = 156238304129265627L;
    public static final double DISCRETIZE_ECU_UNIT = 0.5;
    public static final double MAX_ECU = VM.ECU[VM.FASTEST];
    public static final double MIN_ECU = 2.5;
    private double ECU_Unit = MIN_ECU; // basis ECU
    private int VMType; //the type of VM
    private boolean isUsed = false; //whether this container will be used for task

    public Container(int VMType) {
        this.VMType = VMType;
    }

    public Container(double maxSpeed) {
        this.VMType = VM.FASTEST;
        this.ECU_Unit = (int) (maxSpeed) + 1;
    }

    public void increaseECU() { ECU_Unit = Math.min(ECU_Unit + 0.5, MAX_ECU);}

    public void decreaseECU() { ECU_Unit = Math.max(ECU_Unit - 0.5, MIN_ECU);}

    public double getECU() {
        return ECU_Unit;
    }

    public double getVMCostSingleFit() {
        return VM.ECU[VMType] * VM.UNIT_COSTS[VMType];
    }
    public double getUnitCost() {
        return (getECU() / VM.ECU[VMType]) * VM.UNIT_COSTS[VMType];
    }
}
