import org.junit.Test;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.*;

import static weka.core.Utils.joinOptions;

public class Find {
    @Test
    public void Test() throws Exception {
        String data_dir="E:\\study\\master1-1\\CSVdataset\\";
        String file = "Abalone-3.CSV";

        CSVLoader loader = new CSVLoader();
        loader.setFile(new File(data_dir + file));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);


        CVParameterSelection cv = new CVParameterSelection();
//            String[] options = new String[2];
//            options[0] = "-W";
//            options[1] = "weka.classifiers.trees.J48";
        cv.setClassifier(new J48());
        cv.setNumFolds(10);
        cv.addCVParameter("M 3 10 8");//叶节点最小
        cv.addCVParameter("C 0.1 0.5 9");

        cv.buildClassifier(data);   // build classifier
        Map<String, String> resultMap = new HashMap();
        resultMap.put("bestClassifierOptions", joinOptions(cv.getBestClassifierOptions()));
        System.out.println(resultMap.toString());

    }
}
