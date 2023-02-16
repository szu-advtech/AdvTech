package com.homework.apriori;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class AprioriPass1Mapper extends Mapper<Object, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text number = new Text();

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        //为啥是两个\  因为\\ => \ 从而\\s => \s
        String[] ids = value.toString().split("[\\s\\t]+");
        for (int i = 0; i < ids.length; i++) {
            context.write(new Text(ids[i]), one);
        }
    }
}
