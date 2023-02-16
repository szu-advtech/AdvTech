import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.*;
import java.util.Set;
import java.util.TreeSet;


public class classifier {
	public static Object load(String FilePath){
    	Object o = null;
        try {
        	ObjectInputStream in = new ObjectInputStream(new FileInputStream(FilePath));
            o =in.readObject();
            in.close();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
        return o;
    }
	
	public static double mean(List<Double>l){
		if(l.size()==0){
			return 0;
		}
		double sum=0;
		for(Double i:l){
			sum+=i;
		}
		return sum/l.size();
	}
	
	public static double geomean(List<Double>l){
		if(l.size()==0){
			return 0;
		}
		double mean=0;
		for(Double i:l){
			mean+=Math.log10(i);
		}
		mean/=l.size();
		return Math.pow(10, mean);
	}
	
	public static double max(List<Double>l){
		double max=0;
		for(Double i:l){
			if(i>max)
				max=i;
		}
		return max;
	}
	
	public static double min(List<Double>l){
		double min=1;
		for(Double i:l){
			if(i<min)
				min=i;
		}
		return min;
	}
	
	public static double trimmean(List<Double>l){
		double sum=0;
		for(Double i:l){
			sum+=i;
		}
		if(l.size()<4){
			return mean(l);
		}
		double max=0,min=1;
		for(Double i:l){
			if(i>max)
				max=i;
			if(i<min)
				min=i;
		}
		return (sum-max-min)/(l.size()-2);
	}
	
	public static double middle(List<Double>l){
		if(l.size()==0)
			return 0;
		Collections.sort(l);
		
		if(l.size()%2==0)
			return (l.get(l.size()/2)+l.get(l.size()/2-1));
		else 
			return l.get(l.size()/2);
	}
	
	public static void main(String args[]){
		boolean verbose= args.length>1;
		String patch_no=args[0];
		Set<Integer> pass=(Set<Integer>) load(patch_no+"/pass"); 
		Set<Integer> gen=(Set<Integer>) load(patch_no+"/gen");
		Set<Integer> fail=(Set<Integer>) load(patch_no+"/fail");
		String[] dict=(String[]) load(patch_no+"/dict");
		double[][] dis=(double[][]) load(patch_no+"/dis");
		double[] dis_2=(double[]) load(patch_no+"/dis_2");
		int len=dis_2.length;
        if(verbose){
            for(int i=0;i<len;i++){
                        if(!(pass.contains(i)||gen.contains(i)||fail.contains(i))){
                                continue;
                        }
                        for(int j=0;j<len;j++){
                                if(!(pass.contains(i)||gen.contains(i)||fail.contains(i))){
                                        continue;
                                }
                                System.out.printf("%.6f ",dis[i][j]);
                        }
                        System.out.println();
                }
                System.out.println();
        }

        if(verbose){
                for(int j=0;j<len;j++){
                        if(!(pass.contains(j)||gen.contains(j)||fail.contains(j))){
                                continue;
                        }
                        System.out.printf("%.6f ",dis_2[j]);
                }
                System.out.println();
                for(int j=0;j<len;j++){
                        if(!(pass.contains(j)||gen.contains(j)||fail.contains(j))){
                                continue;
                        }
                        if(pass.contains(j))
                                System.out.print("    pass ");
                        if(fail.contains(j))
                                System.out.print("    fail ");
                        if(gen.contains(j))
                                System.out.print("     gen ");
                }
                System.out.println();
        }
        if(false){
        for (int i:pass){
            for(int j:fail){
                System.out.println(dis[i][j]);
            }
        }
        System.out.println();
        for (int i:pass){
            for(int j:pass){
                if(i<=j)
                    continue;
                System.out.println(dis[i][j]);
            }
        }
        System.out.println();
        for(int i:fail){
            for (int j:fail){
                if(i<=j)
                    continue;
                System.out.println(dis[i][j]);
            }
        }
        }
        List<Double> pass_distances=new ArrayList<Double>();
		List<Double> fail_distances=new ArrayList<Double>();
		if(true){
			for(int i:pass){
				pass_distances.add(dis_2[i]);
				//dis_pass_max=Math.max(dis_pass_max, dis_2[i]);
			}
			for(int i:fail){
				fail_distances.add(dis_2[i]);
			}
			if(gen.size()!=0){
				for(int i:gen){
					double dis_p=1,dis_f=1;
					double dis_p_aver=0,dis_f_aver=0;
					for(int j:pass){
						if(dis_2[j]<dis_p)
							dis_p=dis[i][j];
						dis_p_aver+=dis[i][j];
					}
					dis_p_aver/=pass.size();
					for(int j:fail){
						if(dis_2[j]<dis_f)
							dis_f=dis[i][j];
						dis_f_aver+=dis[i][j];
						
					}
					dis_f_aver/=fail.size();
					if(pass.size()!=0){
						//useless
						if(dis_p<dis_f){
							//pass
							pass_distances.add(dis_2[i]);
							
						} else if(dis_p>dis_f) {
							fail_distances.add(dis_2[i]);
						}
					} else {
						if(dis_f >= 0.4) {
							
							pass_distances.add(dis_2[i]);
						}
					}
					
				}
			}
		}
		
		
        double dis_pass,dis_fail;
		dis_pass=max(pass_distances);
		dis_fail=mean(fail_distances);
                if(dis_pass>=0.25 || dis_pass>dis_fail){
                        System.out.println("Incorrect");
                } else System.out.println("Correct");		
                //System.out.println(fail_distances.size()+pass_distances.size());
	}
}
