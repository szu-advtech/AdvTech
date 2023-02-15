function Res=my_initialRes

    
Res.svmAcc=0; 
Res.svmSen=0; 
Res.svmSpec=0; 
Res.svmPrec=0; 
Res.svmFscore=0; 
Res.svmAuc=0; 

Res.maxAcc=0;
Res.minArmse=8888;  % 越小越好

Res.maxBcc=0;
Res.minBrmse=8888;

Res.showLine1='-------------------------------';
    
Res.svmAcc_Std=0; 
Res.svmSen_Std=0; 
Res.svmSpec_Std=0; 
Res.svmPrec_Std=0; 
Res.svmFscore_Std=0; 
Res.svmAuc_Std=0; 

Res.maxAcc_Std=0;
Res.minArmse_Std=0;  

Res.maxBcc_Std=0;
Res.minBrmse_Std=0;


Res.showLine2='-------------------------------';



end









