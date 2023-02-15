package com.homework.apriori;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Apriori extends Configured implements Tool {

    public static int s;
    public static int k;

    public int run(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();

        String hdfsInputDir = args[0];
        String hdfsOutputDirPrefix = args[1];

        s = Integer.parseInt(args[2]); //阈值
        k = Integer.parseInt(args[3]); //k次迭代,k次pass

        //循环执行pass
        for (int pass = 1; pass <= k; pass++) {
            long passStartTime = System.currentTimeMillis();

            //配置执行job
            if (!runPassKMRJob(hdfsInputDir, hdfsOutputDirPrefix, pass)) {
                return -1;
            }

            long passEndTime = System.currentTimeMillis();
            System.out.println("pass " + pass + " time : " + (passEndTime - passStartTime));
        }

        long endTime = System.currentTimeMillis();
        System.out.println("total time : " + (endTime - startTime));

        return 0;
    }

    private static boolean runPassKMRJob(String hdfsInputDir, String hdfsOutputDirPrefix, int passNum)
            throws IOException, InterruptedException, ClassNotFoundException {
        Configuration passNumMRConf = new Configuration();
        passNumMRConf.setInt("passNum", passNum);
        passNumMRConf.set("hdfsOutputDirPrefix", hdfsOutputDirPrefix);
        passNumMRConf.setInt("minSup", s);
        //加一个不频繁的输出试试
        //passNumMRConf.set("infrequentList", "TestInit");

        Job passNumMRJob = new Job(passNumMRConf, "" + passNum);
        passNumMRJob.setJarByClass(Apriori.class);
        if (passNum == 1) {
            passNumMRJob.setMapperClass(AprioriPass1Mapper.class);
        } else {
            passNumMRJob.setMapperClass(AprioripassKMapper.class);
        }
        passNumMRJob.setReducerClass(AprioriReducer.class);

        passNumMRJob.setOutputKeyClass(Text.class);
        passNumMRJob.setOutputValueClass(IntWritable.class);

        //设置相应数量的ReduceTask
        passNumMRJob.setNumReduceTasks(1);

        FileInputFormat.addInputPath(passNumMRJob, new Path(hdfsInputDir));
        FileOutputFormat.setOutputPath(passNumMRJob, new Path(hdfsOutputDirPrefix + passNum));

        boolean res = passNumMRJob.waitForCompletion(true);

        //String infrequentList = passNumMRConf.get("infrequentList");
        //System.out.println("infrequentList = " + infrequentList);

        return res;

    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new Apriori(), args);
        System.exit(exitCode);
    }

}
