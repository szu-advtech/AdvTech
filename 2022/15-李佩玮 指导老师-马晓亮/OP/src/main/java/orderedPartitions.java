import weka.classifiers.meta.OrdinalClassClassifier;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.Date;
import java.util.Random;


public class orderedPartitions {
    public static void main(String[] args) throws Exception {
        String data_dir="E:\\study\\master1-1\\CSVdataset\\";
        String file = "Abalone-3.CSV";
        CSVLoader loader = new CSVLoader();


        loader.setFile(new File(data_dir + file));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        double sum = 0;
        for(int i=0;i<10; i++){
            OrdinalClassClassifier tree = new OrdinalClassClassifier();
            String[] options = new String[7];
            options[0] = "-W";
            options[1] = "weka.classifiers.trees.J48";
            options[2] = "--";
            options[3] = "-C";
            options[4] = "0.25";
            options[5] = "-M";
            options[6] = "7";
            tree.setOptions(options);
            tree.buildClassifier(data);   // build classifier
            Evaluation eval = new Evaluation(data);
            Date date = new Date();
            Long seed = Long.parseLong(String.format("%tN", date));

            eval.crossValidateModel(tree, data, 10, new Random(seed));
            //System.out.println(eval.toSummaryString());// 输出总结信息
            //System.out.println(eval.errorRate());// 输出错误率
            double Correct_rate = 1-eval.errorRate();
            sum += Correct_rate;
        }
        System.out.printf("10次十折交叉验证的平均正确率为"+String.format("%.2f",sum*10) + "%%");
    }
}
