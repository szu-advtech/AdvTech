import org.apache.commons.lang3.SerializationUtils;

import java.util.*;
import java.util.stream.Collectors;

public class Fitter {
    public static HashMap<VM, List<Container>> fitMapping = new LinkedHashMap<>();

    /**
     *
     * @param vms the vm set that will be select by the best fit
     * @param container the container that should be stored in the VM
     * @param flag whether image should be no considered (true: do not consider the image, false: consider the image)
     * @return the best fit one
     */
    public static VM bestFit(List<VM> vms, Container container, boolean flag) {
        double needECU = container.getECU();
        List<VM> vmList = vms.stream().filter(vm -> flag || vm.getImages().contains(container))
                .sorted((o1, o2) -> (int) (o1.getResidueECU() - o2.getResidueECU())).collect(Collectors.toList());
        int low = 0;
        int high = vmList.size()-1;
        while (low < high) {
            int mid = (high+low)/2;
            double residueECU = vmList.get(mid).getResidueECU();
            if (Math.abs(needECU - residueECU) < 0.0001) {
                break;
            }
            else if (needECU < residueECU) high = mid - 1;
            else if (needECU > residueECU) low = mid + 1;
        }
        if (low >= vmList.size()) return null;

        VM vm = vmList.get(low);
        if (vm.getResidueECU() < container.getECU()) return null; //have not residue ECU
        else {
            container.setVMType(vm.getType());
            vm.setResidueECU(vm.getResidueECU() - container.getECU());
            vm.getLoadedContainer().add(container);
            return vm;
        }
    }

    /**
     *
     * @param vms the new created VM
     * @param VMType the next VMType need to fit
     * @param newIns the Container should be fit
     * @return the VM fit situation of current strategy
     */
    public static HashMap<VM, List<Container>> FFD(List<VM> vms, int VMType, List<Container> newIns) {
        for (Container container : newIns) {
            boolean fitted = false;
            VM fittedVM = null;
            for (VM vm : vms) {
                if (container.getECU() < vm.getResidueECU()) {
                    fittedVM = vm;
                    container.setVMType(vm.getType());
                    vm.setResidueECU(vm.getResidueECU() - container.getECU());
                    fitted = true;
                    break;
                }
            }
            //need to create a new VM to fit this
            if (!fitted) {
                VM newCreateVM = new VM(VMType);
                vms.add(newCreateVM);
                container.setVMType(VMType);
                newCreateVM.setResidueECU(newCreateVM.getResidueECU() - container.getECU());
                fittedVM = newCreateVM;
            }

            List<Container> containerList = fitMapping.computeIfAbsent(fittedVM, k -> new ArrayList<>());
            containerList.add(container);
        }

        //deep clone, because the map will not change, but the VM/Container object's value will change
        return SerializationUtils.clone(fitMapping);
    }

    /**
     *
     * @param fitMapping the configuration with the min cost
     */
    public static void reFitVM(HashMap<VM, List<Container>> fitMapping) {
        HashMap<VM, VM> modify = new LinkedHashMap<>();
        for (Map.Entry<VM, List<Container>> entry : fitMapping.entrySet()) {
            VM vm = entry.getKey();
            double occupyECU = vm.getECU() - vm.getResidueECU();
            int VMType = -1;
            for (int i = vm.getType()-1; i>=0; i--) {
                double ecu = VM.ECU[i];
                if (ecu > occupyECU || Math.abs(occupyECU - ecu) < 1e-6) VMType = i;
                else break;
            }
            //change the lowest cost and satisfy one
            if (VMType != -1) {
                VM newCreateVM = new VM(VMType);
                List<Container> containers = entry.getValue();
                newCreateVM.setResidueECU(newCreateVM.getResidueECU()- occupyECU);
                for (Container container : containers) container.setVMType(VMType);
                modify.put(vm, newCreateVM);
            }
        }
        for (Map.Entry<VM, VM> entry : modify.entrySet()) {
            VM oldVM = entry.getKey();
            VM newVM = entry.getValue();
            List<Container> containers = fitMapping.get(oldVM);
            fitMapping.remove(oldVM);
            fitMapping.put(newVM, containers);
        }

        //set the VM's loadedContainer
        for (Map.Entry<VM, List<Container>> entry : fitMapping.entrySet()) {
            VM vm = entry.getKey();
            List<Container> containers = entry.getValue();
            vm.getLoadedContainer().addAll(containers);
        }
    }
}
