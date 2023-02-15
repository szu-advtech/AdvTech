package org.yaodong.util;

import org.junit.Test;
import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;
public class test111 {
    @Test
    public void test(){
        BloomFilter.create(Funnels.stringFunnel("utf-8"), );
        BloomFilter bloomFilter = BloomFilter.create(Funnels.integerFunnel(), 100000, 0.01);
        for(int i=0; i<100000; i++)
            bloomFilter.put(i);
        System.out.println(bloomFilter.mightContain(1));

        System.out.println(bloomFilter.mightContain(2));

        System.out.println(bloomFilter.mightContain(3));

        System.out.println(bloomFilter.mightContain(100001));
    }
}
