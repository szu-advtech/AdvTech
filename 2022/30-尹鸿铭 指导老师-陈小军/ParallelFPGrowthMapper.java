/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.fpm.pfpgrowth;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;

import org.apache.mahout.common.Parameters;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.apache.mahout.math.set.OpenIntHashSet;

/**
 *  maps each transaction to all unique items groups in the transaction. mapper
 * outputs the group id as key and the transaction as value
 * 
 */
public class ParallelFPGrowthMapper extends Mapper<LongWritable,Text,IntWritable,TransactionTree> {

  private final OpenObjectIntHashMap<String> fMap = new OpenObjectIntHashMap<String>();
  private Pattern splitter;
  private int maxPerGroup;

  private IntWritable wGroupID = new IntWritable();

  @Override
  protected void map(LongWritable offset, Text input, Context context)
    throws IOException, InterruptedException {

    String[] items = splitter.split(input.toString());

    OpenIntHashSet itemSet = new OpenIntHashSet();

    for (String item : items) {
      if (fMap.containsKey(item) && !item.trim().isEmpty()) {
        itemSet.add(fMap.get(item));
      }
    }

    IntArrayList itemArr = new IntArrayList(itemSet.size());
    itemSet.keys(itemArr);
    itemArr.sort();

    OpenIntHashSet groups = new OpenIntHashSet();
    for (int j = itemArr.size() - 1; j >= 0; j--) {
      // generate group dependent shards
      int item = itemArr.get(j);
      int groupID = PFPGrowth.getGroup(item, maxPerGroup);
        
      if (!groups.contains(groupID)) {
        IntArrayList tempItems = new IntArrayList(j + 1);
        tempItems.addAllOfFromTo(itemArr, 0, j);
        context.setStatus("Parallel FPGrowth: Generating Group Dependent transactions for: " + item);
        wGroupID.set(groupID);
        context.write(wGroupID, new TransactionTree(tempItems, 1L));
      }
      groups.add(groupID);
    }
    
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    int i = 0;
    for (Pair<String,Long> e : PFPGrowth.readFList(context.getConfiguration())) {
      fMap.put(e.getFirst(), i++);
    }
    
    Parameters params = 
      new Parameters(context.getConfiguration().get(PFPGrowth.PFP_PARAMETERS, ""));

    splitter = Pattern.compile(params.get(PFPGrowth.SPLIT_PATTERN,
                                          PFPGrowth.SPLITTER.toString()));
    
    maxPerGroup = Integer.valueOf(params.getInt(PFPGrowth.MAX_PER_GROUP, 0));
  }


//  //构建特征集合矩阵
//  protected static String[][] characteristicMatrix(Set<String> set, Set<String> set1, Set<String> set2){
//    String[] a = new String[set.size()];
//    set.toArray(a);
//    String[] set1Array = new String[set1.size()];
//    set1.toArray(set1Array);
//    String[] set2Array = new String[set2.size()];
//    set2.toArray(set2Array);
//    String[][] matrix = new String[a.length][5];//此处构造为5是为了后面的最小哈希签名中的两个哈希函数的结果存放。
//    int i, j, temp;
//    for(i = 0; i < matrix.length; i++){
//      for(j = 0; j < matrix[0].length; j++){
//        matrix[i][j] = "0";
//      }
//    }
//    i = 0;
//    for(Iterator<String> iter = set.iterator(); iter.hasNext();){
//      matrix[i++][0] = iter.next();
//    }
//    i = 0;
//    temp = 0;
//    for(j = i; j < a.length && temp < set1Array.length; j++){
//      if(matrix[j][0].equals(set1Array[temp])){
//        matrix[j][1] = "1";
//        temp++;
//      }
//    }
//    temp = 0;
//    for(j = i; j < a.length && temp < set2Array.length; j++){
//      if(matrix[j][0].equals(set2Array[temp])){
//        matrix[j][2] = "1";
//        temp++;
//      }
//    }
//    return matrix;
//  }
//
//  //行打乱
//  protected static String[][] rowMess(String[][] matrix){
//    int rowNumber1, rowNumber2;
//    int i, j;
//    String temp;
//    Random r = new Random();
//    //随机进行行打乱十次
//    for(i = 0; i < 9; i++){
//      rowNumber1 = r.nextInt(matrix.length);
//      rowNumber2 = r.nextInt(matrix.length);
//      for(j = 0; j < matrix[0].length; j ++){
//        temp = matrix[rowNumber2][j];
//        matrix[rowNumber2][j] = matrix[rowNumber1][j];
//        matrix[rowNumber1][j] = temp;
//      }
//    }
//    return matrix;
//  }
//
//  //根据最小hash值求相似度
//  protected static double minHashJaccard(int k, Set<String> set) throws IOException{
//    Set<String> set1 = getSet(k, KShingle.ReadFile1());
//    Set<String> set2 = getSet(k, KShingle.ReadFile2());
//    String[][] matrix = characteristicMatrix(set, set1, set2);
//    matrix = rowMess(matrix);
//    double result;
//    System.out.println("已知定义：两个集合经随机排列转换之后得到的两个最小哈希值相等的概率等于这两个集合的jaccard相似度");
//    int equalHashValue = 0;
//    for(int i = 0; i < matrix.length; i++){
//      if(matrix[i][1].equals(matrix[i][2]) && matrix[i][1].equals("1")){
//        equalHashValue++;
//      }
//    }
//    System.out.println("全集共有项的数目：" + set.size());
//    System.out.println("都为1(该子串在两段文本中均出现)的数目：" + equalHashValue);
//    result = equalHashValue * 1.0 / set.size();
//    DecimalFormat df = new DecimalFormat("0.00");
//    System.out.println("第一项与第二项得到最小哈希值相等的概率计算为P = " + equalHashValue + " / "  + set.size() + " = " + df.format(result));
//    System.out.println("即MinHash算得的两段文本之间的jaccard相似度结果为：" + df.format(result));
//    return equalHashValue;
//  }


}
