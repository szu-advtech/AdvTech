import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.util.*;

@Getter
@Setter
@AllArgsConstructor
public class IFFD {
    private List<Container> newIns;
    private List<VM> vms;
    private Configuration conf;

    /**
     *
     * @return the final configuration
     */
    public void execute() {
        List<Container> removeNewIns = new ArrayList<>();
        for (Container container : newIns) {
            VM fitSuccess = Fitter.bestFit(vms, container, true);
            if (fitSuccess != null) {
                removeNewIns.add(container);
                conf.addVM(fitSuccess);
            }
        }
        for (Container container : removeNewIns) newIns.remove(container);

        newIns.sort((n1, n2) -> (int) (n2.getECU() - n1.getECU()));
        vms.sort((o1, o2) -> (int) (o2.getResidueECU() - o1.getResidueECU()));

        List<VM> newCreatedVMs = new LinkedList<>();
        //the first fit
        HashMap<VM, List<Container>> currentB = Fitter.FFD(newCreatedVMs, VM.FASTEST, newIns);
        //should be reFit
        List<Container> containersOfLatestVM = CalHandler.getLatestVMsContainer(currentB);

        HashMap<VM, List<Container>> bestOne = currentB;
        double minCost = CalHandler.calTotalCost(currentB.keySet());

        for (int i=VM.FASTEST-1; i>=0; i--) {
            newCreatedVMs.remove(newCreatedVMs.size()-1); //remove the latest one for replacing of the new VM's type

            currentB = Fitter.FFD(newCreatedVMs, i, containersOfLatestVM);
            containersOfLatestVM = CalHandler.getLatestVMsContainer(currentB);

            double currentCost = CalHandler.calTotalCost(currentB.keySet());
            if (currentCost < minCost) {
                bestOne = currentB;
                minCost = currentCost;
            }
        }

        Fitter.reFitVM(bestOne);
        conf.addVMSet(bestOne.keySet());
    }
}