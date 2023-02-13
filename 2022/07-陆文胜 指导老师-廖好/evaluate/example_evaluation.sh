#python convert_to_rouge.py --ref_data_dir ../City_4_logs/explanation --gen_data_dir ../City_4_logs/explanation --test_number 746
#ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > City_4_rouge_result.txt None

#python convert_to_rouge.py --ref_data_dir ../City_2_logs/explanation --gen_data_dir ../City_2_logs/explanation --test_number 2557
#ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > City_2_rouge_result.txt None

#python convert_to_rouge.py --ref_data_dir ../yelp_log/NV_log/explanation --gen_data_dir ../yelp_log/NV_log/explanation --test_number 1756
#ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > NV_2019data_v5_rouge_result.txt None


#python convert_to_rouge.py --ref_data_dir ../City2_log/16:01:23:45:27/explanation --gen_data_dir ../City2_log/16:01:23:45:27/explanation --test_number 1656
#ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > dianping_City2_16:01:23:45:27_rouge_result.txt None

#python convert_to_rouge.py --ref_data_dir ../ERCP_C_NV_log/08:10:02:17:30/explanation --gen_data_dir ../ERCP_C_NV_log/08:10:02:17:30/explanation --test_number 1756
#ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > NV_08:10:02:17:30_rouge_result.txt None


#python convert_to_rouge.py --ref_data_dir ../ERCP_C_City4_log/29:08:10:46:40/explanation --gen_data_dir ../ERCP_C_City4_log/29:08:10:46:40/explanation --test_number 746
#ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > City4_29:08:10:46:40_rouge_result.txt None


#python convert_to_rouge.py --ref_data_dir ../ERCP_C_City2_log/29:08:10:53:39/explanation --gen_data_dir ../ERCP_C_City2_log/29:08:10:53:39/explanation --test_number 1656
#ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > City2_29:08:10:53:39:2_rouge_result.txt None

python convert_to_rouge.py --ref_data_dir ../City4_log/dianping/RAW_MSE_CAML_FN_FM/City4/expalanation --gen_data_dir ../City4_log/dianping/RAW_MSE_CAML_FN_FM/City4/expalanation
ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f B -p 0.5 -t 0 config.xml > AZ_09:10:12:59:09_rouge_result.txt None
