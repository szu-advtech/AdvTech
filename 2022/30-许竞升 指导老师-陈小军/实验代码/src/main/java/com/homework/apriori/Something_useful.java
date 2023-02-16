package com.homework.apriori;

public class Something_useful {
    /*
    Mapper有setup()，map()，cleanup()和run()四个方法。其中setup()一般是用来进行一些map()前的准备工作，map()则一般承担主要的处理工作，
    cleanup()则是收尾工作如关闭文件或者执行map()后的K-V分发等。run()方法提供了setup->map->cleanup()的执行模板。

    在MapReduce中，Mapper从一个输入分片中读取数据，然后经过Shuffle and Sort阶段，分发数据给Reducer，在Map端和Reduce端我们可能使用
    设置的Combiner进行合并，这在Reduce前进行。Partitioner控制每个K-V对应该被分发到哪个reducer[我们的Job可能有多个reducer]，Hadoop
    默认使用HashPartitioner，HashPartitioner使用key的hashCode对reducer的数量取模得来。

    https://www.cnblogs.com/xuepei/p/3607109.html#:~:text=Mapper%E7%B1%BB4%E4%B8%AA%E5%87%BD%E6%95%B0%E7%9A%84%E8%A7%A3%E6%9E%90%20Mapper%E6%9C%89setup%20%28%29%EF%BC%8Cmap%20%28%29%EF%BC%8Ccleanup%20%28%29%E5%92%8Crun%20%28%29%E5%9B%9B%E4%B8%AA%E6%96%B9%E6%B3%95%E3%80%82,%E5%85%B6%E4%B8%ADsetup%20%28%29%E4%B8%80%E8%88%AC%E6%98%AF%E7%94%A8%E6%9D%A5%E8%BF%9B%E8%A1%8C%E4%B8%80%E4%BA%9Bmap%20%28%29%E5%89%8D%E7%9A%84%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C%EF%BC%8Cmap%20%28%29%E5%88%99%E4%B8%80%E8%88%AC%E6%89%BF%E6%8B%85%E4%B8%BB%E8%A6%81%E7%9A%84%E5%A4%84%E7%90%86%E5%B7%A5%E4%BD%9C%EF%BC%8Ccleanup%20%28%29%E5%88%99%E6%98%AF%E6%94%B6%E5%B0%BE%E5%B7%A5%E4%BD%9C%E5%A6%82%E5%85%B3%E9%97%AD%E6%96%87%E4%BB%B6%E6%88%96%E8%80%85%E6%89%A7%E8%A1%8Cmap%20%28%29%E5%90%8E%E7%9A%84K-V%E5%88%86%E5%8F%91%E7%AD%89%E3%80%82
     */
}
