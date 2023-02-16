import org.junit.Test;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.MultiClassClassifier;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.Date;
import java.util.Random;

public class Disordered {
    @Test
    public void Test() throws Exception {
        String data_dir="E:\\study\\master1-1\\CSVdataset\\";
        String file = "Abalone-3.CSV";

        CSVLoader loader = new CSVLoader();
        loader.setFile(new File(data_dir + file));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        double sum = 0;
        for(int i=0;i<10; i++) {
            String[] options = new String[2];
            options[0] = "-M";
            options[1] = "0";
            MultiClassClassifier tree = new MultiClassClassifier();
            tree.setOptions(options);

            J48 j48 = new J48();
            String[] options2 = new String[2];
            options2[0] = "-M";
            options2[1] = "7";
            j48.setOptions(options2);

            tree.setClassifier(j48);
            tree.buildClassifier(data);   // build classifier

            Evaluation eval = new Evaluation(data);

            Date date = new Date();
            Long seed = Long.parseLong(String.format("%tN", date));

            eval.crossValidateModel(tree, data, 10, new Random(seed));
            //System.out.println(eval.toSummaryString());// 输出总结信息
            //System.out.println(eval.toClassDetailsString());// 输出分类详细信息

            double Correct_rate = 1-eval.errorRate();
            sum += Correct_rate;
        }
        System.out.printf("10次十折交叉验证的平均正确率为"+String.format("%.2f",sum*10) + "%%");
    }
}
