package com.homework.apriori;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.*;

public class AprioriReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();
//    String infrequent = "";

    //他这里只共用了一个reducer，但是复现论文中应该使用3个ruducer
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum = 0;
        int minSup = context.getConfiguration().getInt("minSup", 5);
        for (IntWritable val : values) {
            sum += val.get();
        }
        result.set(sum);

        if (sum > minSup) {
            context.write(key, result);
        }
//        else {
//            infrequent += key + "|";
//        }
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
        //为什么集群中写文件不能用这两段代码呢

//        String infrequentList = context.getConfiguration().get("infrequentList"," ");
//        infrequentList = infrequentList.substring(0, infrequentList.length() - 1);
//        FileWriter writer;
//        try {
//            writer = new FileWriter("/user/xujingsheng/infrequent.txt");
//            writer.write(""); //清空原文件内容
//            writer.write(infrequentList);
//            writer.write("asdfg");
//            writer.flush();
//            writer.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

//        int passNum = context.getConfiguration().getInt("passNum", 2);
//        String location = context.getConfiguration().get("fs.default.name") + "/user/xujingsheng/infrequent2.txt";
//
//        File file = new File(location);
//
//        if(!file.exists()){
//            file.createNewFile();
//        }
//
//        FileWriter fw = new FileWriter(file.getAbsoluteFile());
//        BufferedWriter bw = new BufferedWriter(fw);
//        bw.write("456");
//        bw.close();

        /*
        String location = "/user/xujingsheng/infrequent.txt";

        Path path = new Path(location);
        FileSystem fs = FileSystem.get(context.getConfiguration());
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs.create(path)));
        if (infrequent.length() != 0) {
            infrequent = infrequent.substring(0, infrequent.length() - 1);
        }

        //bw.write(infrequent + "  " + context.getConfiguration().get("fs.default.name"));
        bw.write(infrequent);
        bw.close();
         */
    }
}
