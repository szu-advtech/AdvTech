package com.homework.apriori;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class AprioripassKMapper_Top extends Mapper<Object, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text item = new Text();

    private List<List<Integer>> prevItemsets = new ArrayList<List<Integer>>();
    private List<List<Integer>> candidateItemsets = new ArrayList<List<Integer>>();
    private Map<String, Boolean> candidateItemsetsMap = new HashMap<String, Boolean>();
    private List<List<Integer>> infrequentItemsets = new ArrayList<List<Integer>>();

    //第一个以后的pass使用该Mapper，在map函数执行前会执行setup来从k-1次pass的输出中构建候选itemsets,对应于apriori算法
    @Override
    public void setup(Context context) throws IOException, InterruptedException {
        int passNum = context.getConfiguration().getInt("passNum", 2);
        String prefix = context.getConfiguration().get("hdfsOutputDirPrefix", "");
        String lastPass1 = context.getConfiguration().get("fs.default.name") + "/user/hadoop/chess-" + (passNum - 1) + "/part-r-00000";
        String lastPass = context.getConfiguration().get("fs.default.name") + prefix + (passNum - 1) + "/part-r-00000";
        String infrequentPath = context.getConfiguration().get("fs.default.name") + "/user/xujingsheng/infrequent.txt";
        String datasetsPathString = context.getConfiguration().get("fs.default.name") + "/user/xujingsheng/dataset/" + passNum;

        //读入不频繁项集
        FileSystem fs = FileSystem.get(context.getConfiguration());
        Path infrequentloc = new Path(infrequentPath);
        BufferedReader ibf = new BufferedReader(new InputStreamReader(fs.open(infrequentloc)));
        String infrequentLine = ibf.readLine();
        String[] infrequentStringList = infrequentLine.split("|");

        //把不频繁项集弄成数组形式
        List<Integer> infrequent_itemset = new ArrayList<Integer>();
        for (String infrequentString : infrequentStringList) {
            List<Integer> itemset = new ArrayList<Integer>();
            for (String itemStr : infrequentString.split(",")) {
                itemset.add(Integer.parseInt(itemStr));
            }
            infrequentItemsets.add(itemset);
        }

        //get candidate itemsets from the removed prevItemsets
        //跟新了删去不频繁项集的原始dataset之后我们又需要重新写个文件
        //读入先前的数据集
        Path datasetsPath = new Path(datasetsPathString);
        BufferedReader dbf = new BufferedReader(new InputStreamReader(fs.open(datasetsPath)));
        String line = null;
        while ((line = dbf.readLine()) != null) {
            List<Integer> transaction = new ArrayList<Integer>();
            for (String item : line.split("[\\s\\t]+")) {
                transaction.add(Integer.parseInt(item));
            }
            //删掉每一行中存在的不频繁项集的子集
            List<Integer> removeNum = new ArrayList<Integer>();
            for (List<Integer> infrequent : infrequentItemsets) {
                contains_Top(infrequent, transaction, removeNum);
            }
            Collections.sort(removeNum);
            int temp = -1;
            for (int i = removeNum.size() - 1; i >= 0; i--) {
                if (temp != removeNum.get(i)) {
                    transaction.remove(removeNum.get(i));
                    temp = removeNum.get(i);
                }
            }

        }


    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        //意思就是说要把一开始的文本表示的项集弄成ArrayList形式，项集以行为单位存在块上（可能ArrayList保存的时候就带了空格啥的）
        //这个value每次读入的都是最初始的input文件
        //感觉这里的 itemset 得改名为 transaction
        //所以每进行一次map就是对一个事务进行一次遍历，判断由k-1项集产生的候选集k项集是不是满足事务，满足则+1，不满足就丢了
        String[] ids = value.toString().split("[\\s\\t]+");
        List<Integer> itemset = new ArrayList<Integer>();
        for (String id : ids) {
            itemset.add(Integer.parseInt(id));
        }

        //在这里可以添加论文中对原始数据进行操作的规则（比如说TopAprioriMR中，删去所有含有不频繁项集的事务）
        //跟新了删去不频繁项集的原始dataset之后我们又需要重新写个文件


        //遍历所有候选集合
        for (List<Integer> candidateItemset : candidateItemsets) {
            //意思是看传进来的itemset在不在候选集里面，在的话就写出，不在就丢弃
            //感觉这里的这个顺序是不是有bug
            //好像不是上面的的想法，itemset是事务，这里应该是看cadidateItemset在不在一个事务内，或者是不是一个事务的子集
            //如果是一个事务的子集，则加逗号写入下一栏reducer，若不是则舍弃这个cadidateItemset
            //candidateItemset才是我们每轮所需要讨论的项集，或者说是继上一次生成后产生的新项集
            if (contains(candidateItemset, itemset)) {
                String outputkey = "";
                for (int i = 0; i < candidateItemset.size(); i++) {
                    //把数组又给转成以逗号分隔的文本了
                    //有点没想明白这里为什么加逗号？？？？
                    //中间结果会加上逗号
                    outputkey += candidateItemset.get(i) + ",";
                }
                //这句话是啥意思？？？？
                //去掉末尾的逗号
                outputkey = outputkey.substring(0, outputkey.length() - 1);
                context.write(new Text(outputkey), one);
            }
        }

    }

    //返回items是否是allItems的子集
    private boolean contains(List<Integer> items, List<Integer> allItems) {
        int i = 0;
        int j = 0;
        while (i < items.size() && j < allItems.size()) {
            if (allItems.get(j) > items.get(i)) {
                return false;
            } else if (allItems.get(j) == items.get(i)) {
                i++;
                j++;
            } else {
                j++;
            }
        }
        if (i != items.size()) {
            return false;
        }
        return true;
    }

    private boolean contains_Top(List<Integer> items, List<Integer> allItems, List<Integer> removeNum) {
        int i = 0;
        int j = 0;
        int count = 0;
        while (i < items.size() && j < allItems.size()) {
            if (allItems.get(j) > items.get(i)) {
                for (int k = count; k >= 0; k--) {
                    removeNum.remove(removeNum.size() - 1);
                }
                return false;
            } else if (allItems.get(j) == items.get(i)) {
                removeNum.add(i);
                count++;
                i++;
                j++;
            } else {
                j++;
            }
        }
        if (i != items.size()) {
            for (int k = count; k >= 0; k--) {
                removeNum.remove(removeNum.size() - 1);
            }
            return false;
        }
        return true;
    }

    //获取所有候选集合，参考apriori算法
    private List<List<Integer>> getCandidateItemsets(List<List<Integer>> prevItemsets, int passNum) {
        List<List<Integer>> candidateItemsets = new ArrayList<List<Integer>>();

        //上次pass的输出(prevItemsets)中选取连个itemset构造大小为k + 1的候选集合
        //但这样的情况会出现重复计算 比如 13 + 15 = 135 和 15 + 35 = 135
        //所以重复出现的情况在下面还有个剪枝操作那儿进行了判断
        for (int i = 0; i < prevItemsets.size(); i++) {
            for (int j = i + 1; j < prevItemsets.size(); j++) {
                List<Integer> outerItems = prevItemsets.get(i);
                List<Integer> innerItems = prevItemsets.get(j);

                List<Integer> newItems = null;
                if (passNum == 1) {
                    //如果是第一次的画都是单项集
                    newItems = new ArrayList<Integer>();
                    newItems.add(outerItems.get(0));
                    newItems.add(innerItems.get(0));
                } else {
                    int nDifferent = 0;
                    int index = -1;
                    //比较两个项有多少个不同的地方
                    for (int k = 0; k < passNum; k++) {
                        if (!innerItems.contains(outerItems.get(k))) {
                            nDifferent++;
                            index = k;
                        }
                    }
                    //如果两个项只有1个不同的地方我们就可以生成一个新的k+1项
                    if (nDifferent == 1) {
                        newItems = new ArrayList<Integer>();
                        //这里加谁都一样，反正元素都相同只加一个不同的进去
                        newItems.addAll(innerItems);//addAll是一个一个元素加进去，相当于复制,赋值
                        newItems.add(outerItems.get(index));
                    }
                }
                if (newItems == null) continue; //没有的话就跳过下面的步骤了
                //有的话就要判断是否加入候选集
                Collections.sort(newItems);
                //剪枝：候选集合必须满足所有的子集都在上次pass的输出中，调用 isCandidate 进行检测，通过后加入到候选子集和列表
                //但这里实现的是直接从k-1频繁项集中生成k项集作为候选集，这个候选集可以删去大部分 k-1不频繁项集
                //但也会存在一种情况：即两个k-1项集不同的地方的组合恰好是先前的不频繁项集中的内容
                //因此下面这个isCandidate再一次删去上述的特殊情况
                if (isCandidate(newItems, prevItemsets) && !candidateItemsets.contains(newItems)) {
                    candidateItemsets.add(newItems);
                    //System.out.println(newItems);
                }

            }
        }
        return candidateItemsets;
    }

    //这个简直操作可以看成是 SPAprioriMR 的另一种实现方式
    //算法中提出是保存不频繁项集，从原始事务中生成k项集同时该k项集的某个子集不属于该不频繁项集
    //这里是要求k项候选集的所有子集都是k-1的频繁项集 ———— 不也就可以理解成k项候选集中不存在k-1的不频繁项集
    private boolean isCandidate(List<Integer> newItems, List<List<Integer>> prevItemsets) {
        //用来判断newItems的每个子集是不是都在前一个频繁项集中
        //如果L_k是一个频繁项集，那么其对应的L_(k-1)应该都是频繁项集 —————— 修剪对应的规则
        List<List<Integer>> subsets = getSubsets(newItems);

        for (List<Integer> subset : subsets) {
            if (!prevItemsets.contains(subset)) {
                return false;
            }
        }
        return true;
    }

    private List<List<Integer>> getSubsets(List<Integer> items) {
        List<List<Integer>> subsets = new ArrayList<List<Integer>>();
        for (int i = 0; i < items.size(); i++) {
            List<Integer> subset = new ArrayList<Integer>(items);
            subset.remove(i);
            subsets.add(subset);
        }
        return subsets;
    }

    private void getCandidateItemsets_Top(List<List<Integer>> prevItemsets) {

    }


}
