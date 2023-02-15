import lombok.Data;
import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.helpers.DefaultHandler;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.util.*;

@Data
public class Workflow extends ArrayList<Task> {
    private int maxParallel;
    private double factor; //1.0-2.0
    private double deadline = Double.MAX_VALUE;

    //only used in reading DAX, simulate the data transfer from other job
    private HashMap<String, TransferData> transferData = new HashMap<>();
    private HashMap<String, Task> nameTaskMapping = new HashMap<>();

    public Workflow(String file, double factor) {
        super();
        this.factor = factor;
        Task.resetInternalId();
        //readDAX
        try {
            SAXParser sp = SAXParserFactory.newInstance().newSAXParser();
            sp.parse(new InputSource(file), new MyDAXReader());
            System.out.println("succeed to read DAX data from: " + file);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //add tasks to this workflow
        this.addAll(nameTaskMapping.values());
        Task tentry = new Task(("entry"), 0);
        Task texit = new Task(("exit"), 0);
        //add edges to entry and exit
        for(Task t: this) {
            if(t.getInEdges().size()==0) {
                Edge e = new Edge(tentry, t);
                t.getInEdges().add(e);
                tentry.getOutEdges().add(e);
            }
            if(t.getOutEdges().size()==0) {
                Edge e = new Edge(t, texit);
                t.getOutEdges().add(e);
                texit.getInEdges().add(e);
            }
        }
        //add the entry and exit nodes to the workflows
        this.add(0, tentry);
        this.add(texit);

        //add tasks to this workflow
        bind();
        //topological sort
        selfTopoSort();
        //calculate the value of tasks/WF
        calDDL();
        calSubDDL();
    }

    private void bind(){
        Task tentry = this.get(0);
        Task texit = this.get(this.size() - 1);

        //Bind data flow to control flow
        for(TransferData td : transferData.values()) {
            Task source = td.getSource();
            List<Task> destinations = td.getDestinations();
            //no data source, just set the entry as source, and the size also set to 0
            if(source == null) {
                source = tentry;
                td.setSize(0); //a setting: transfer time of input data is omitted --- setting to 0
            }
            if(destinations == null || destinations.size()==0)	//as same as above
                destinations.add(texit);
            for(Task destination : destinations){
                boolean flag = true;
                for(Edge outEdge : source.getOutEdges()){
                    if(outEdge.getDestination() == destination){
                        outEdge.setDataSize(td.getSize()); //bind here
                        flag = false;
                    }
                }
                //an annoying problem in some DAX files: a data flow cannot be bound to existing control flows
                //flag to indicate whether this problem exists
                if(flag){
                    Edge e = new Edge(source, destination);
                    e.setDataSize(td.getSize());
                    source.insertOutEdge(e);
                    destination.insertInEdge(e);
                    /*System.out.println("**************add a control flow*******************source: "
                            +e.getSource().getName()+"; destination: "+e.getDestination().getName());*/
                }
            }
        }
    }

    private List<Task> topoSort(Task entry) {
        // Empty list that will contain the sorted elements
        List<Task> topoList = new ArrayList<>();
        //Set of all nodes with no incoming edges
        PriorityQueue<Task> S = new PriorityQueue<Task>(10, new Task.ParallelComparator());
        S.add(entry);

        for(Task task : this)	//set topoCount to 0
            task.setTopoCount(0);

        this.maxParallel = -1;
        while(S.size()>0) {
            maxParallel = Math.max(maxParallel, S.size());
            Task task = S.poll(); // remove a node n from S
            topoList.add(task);	// add n to tail of L
            // for each node m with an edge e from n to m do
            for(Edge e : task.getOutEdges()) {
                Task t = e.getDestination();
                t.setTopoCount(t.getTopoCount()+1);	//remove edge e from the graph--achieved by setting topoCount here
                if(t.getTopoCount() == t.getInEdges().size()) //if m has no other incoming edges then
                    S.add(t); // insert m into S
            }
        }

        Edge.EdgeComparator ecForDestination = new Edge.EdgeComparator(true, topoList); //sort edges for each task
        Edge.EdgeComparator ecForSource = new Edge.EdgeComparator(false, topoList);
        //order every in/out edges belong to task by their index of topoSort(from samll to big)
        for(Task t : this) {
            Collections.sort(t.getInEdges(), ecForSource);
            Collections.sort(t.getOutEdges(), ecForDestination);
        }
        return topoList;
    }

    private List<Task> reTopoSort(Task exit) {
        // Empty list that will contain the sorted elements
        List<Task> reTopoList = new ArrayList<>();
        //Set of all nodes with no incoming edges
        PriorityQueue<Task> S = new PriorityQueue<Task>(10, new Task.ParallelComparator());
        S.add(exit);

        for(Task task : this)	//set topoCount to 0
            task.setTopoCount(0);

        this.maxParallel = -1;
        while(S.size()>0) {
            maxParallel = Math.max(maxParallel, S.size());
            Task task = S.poll(); // remove a node n from S
            reTopoList.add(task);	// add n to tail of L
            // for each node m with an edge e from n to m do
            for(Edge e : task.getInEdges()) {
                Task t = e.getSource();
                t.setTopoCount(t.getTopoCount()+1);	//remove edge e from the graph--achieved by setting topoCount here
                if(t.getTopoCount() == t.getOutEdges().size()) //if m has no other outcoming edges then
                    S.add(t); // insert m into S
            }
        }

        Edge.EdgeComparator ecForDestination = new Edge.EdgeComparator(true, reTopoList);
        Edge.EdgeComparator ecForSource = new Edge.EdgeComparator(false, reTopoList);
        //order every in/out edges belong to task by their index of topoSort(from samll to big)
        for(Task t : this) {
            Collections.sort(t.getOutEdges(), ecForDestination);
            Collections.sort(t.getInEdges(), ecForSource);
        }
        return reTopoList;
    }


    // convert the task list of workflow to a topological sort based on Kahn algorithm;
    // besides, calculate maximal parallel number and sort edges for each task
    private void selfTopoSort(){
        Task entry = this.get(0);
        Collections.copy(this, this.topoSort(entry));
    }

    private void calDDL() {
        //calculate from the basis configuration
        double speed = VM.ECU[VM.SLOWEST];

        for(int j=this.size()-1; j>=0; j--) { //find the biggest value(that can decide whether the task have finished)
            // review the handwriting paper
            double bLevel = 0;
            double sLevel = 0;
            Task task = this.get(j);
            for(Edge outEdge : task.getOutEdges()){
                Task child = outEdge.getDestination();
                bLevel = Math.max(bLevel, child.getBLevel() + (double) (outEdge.getDataSize() / VM.NETWORK_SPEED));
                sLevel = Math.max(sLevel, child.getSLevel());
            }
            task.setBLevel(bLevel + (task.getTaskSize() / speed)); //rank(ti)
            task.setSLevel(sLevel + (task.getTaskSize() / speed));
        }

        this.deadline = this.get(0).getBLevel() * factor;
    }

    public void calTaskRankBasedCE(HashMap<Task, Allocation> revMapping){
        //calculate from the container configuration
        for(int j= this.size()-1; j>=0; j--) { //find the biggest value(that can decide whether the task have finished)
            // review the handwriting paper
            double bLevel = 0;
            double sLevel = 0;
            Task task = this.get(j);
            Allocation alloc = revMapping.get(task);
            double speed = alloc.getContainer().getECU();
            for(Edge outEdge : task.getOutEdges()) {
                Task child = outEdge.getDestination();
                bLevel = Math.max(bLevel, child.getBLevel() + (double) (outEdge.getDataSize() / VM.NETWORK_SPEED));
                sLevel = Math.max(sLevel, child.getSLevel());
            }
            task.setRank(bLevel + task.getTaskSize() / speed);
            task.setBLevel(bLevel + task.getTaskSize() / speed); //rank(ti)
            task.setSLevel(sLevel + task.getTaskSize() / speed);
        }
    }

    /*
    the cp must be the one of the outEdges
     */
    public int calHop(Task entry, HashMap<Task, Allocation> revMapping) {
        //calculate the ve
        List<Task> topoList = topoSort(entry);
        for (Task task : topoList) {
            double ve = 0.0;
            List<Edge> inEdges = task.getInEdges();
            for (Edge edge : inEdges) {
                double speed = revMapping.get(task).getContainer().getECU();
                ve = Math.max(ve, edge.getSource().getVe() + (double) (edge.getDataSize() / VM.NETWORK_SPEED) + (task.getTaskSize() / speed));
            }
            task.setVe(ve);
        }

        //calculate the vl
        List<Task> reTopoList = reTopoSort(entry);
        for (Task task : reTopoList) {
            double vl = task.getVe();
            List<Edge> outEdges = task.getOutEdges();
            for (Edge edge : outEdges) {
                double speed = revMapping.get(task).getContainer().getECU();
                vl = Math.min(vl, edge.getDestination().getVl() - ((double) (edge.getDataSize() / VM.NETWORK_SPEED) + (task.getTaskSize() / speed)));
            }
            task.setVl(vl);
        }

        int count = 0;
        //find the task which ve = vl
        for (Task task : topoList)
            if (Math.abs(task.getVe() - task.getVl()) < 1e-6) count++;

        return count == 0 ? 1 : count;
    }

    public void calSubDDL() {
        double speed = VM.ECU[VM.SLOWEST];
        double rankEntry = this.get(0).getBLevel();

        for (int j = this.size()-2; j>0; j--) {
            Task subTask = this.get(j);
            double rank = subTask.getBLevel();
            double ET = subTask.getTaskSize() / speed;
            double sd = deadline * ((rankEntry - rank + ET) / rankEntry);
            subTask.setSubDDL(sd);
        }
    }

    //the DAXReader
    private class MyDAXReader extends DefaultHandler {
        private Stack<String> tags = new Stack<>();
        private String childId;
        private Task lastTask;

        // task->nameTaskMapping td->transferData task.setEdge tags.push(qName)
        public void startElement(String uri, String localName, String qName, Attributes attrs) {
            if(qName.equals("job")){
                String id = attrs.getValue("id");
                if(nameTaskMapping.containsKey(id))		//id conflicts
                    throw new RuntimeException();
                Task t = new Task(id, Double.parseDouble(attrs.getValue("runtime"))); //current job
                nameTaskMapping.put(id, t);
                lastTask = t;
            }else if(qName.equals("uses") && tags.peek().equals("job")){
                //After reading the element "job", the element "uses" means a trasferData (i.e., data flow)
                String filename =attrs.getValue("file");
                long fileSize = Long.parseLong(attrs.getValue("size"));

                TransferData td = transferData.get(filename);
                if(td == null){
                    td = new TransferData(filename, fileSize); //uses file-size
                }

                if(attrs.getValue("link").equals("input")){ //combine with the last task, combine the task
                    td.addDestination(lastTask);
                }else{									//output
                    td.setSource(lastTask);
                }
                transferData.put(filename, td);

            }else if(qName.equals("child") ){
                childId = attrs.getValue("ref");
            }else if(qName.equals("parent") ){
                //After reading the element "child", the element "parent" means an edge (i.e., control flow)
                Task child = nameTaskMapping.get(childId);
                Task parent = nameTaskMapping.get(attrs.getValue("ref"));

                Edge e = new Edge(parent, child);			//control flow
                parent.insertOutEdge(e);
                child.insertInEdge(e);
            }
            tags.push(qName);
        }

        public void endElement(String uri, String localName,String qName) {
            tags.pop();
        }
    }

    //this class is only used in parsing DAX data
    @Data
    private class TransferData{
        private String name;
        private long size;
        private Task source; //used to bind control flow and data flow
        private List<Task> destinations = new ArrayList<>();

        public TransferData(String name, long size) {
            this.name = name;
            this.size = size;
        }

        @Override
        public String toString() {return "TransferData [name=" + name + ", size=" + size + "]";}

        public void addDestination(Task t){destinations.add(t);}
    }
}


