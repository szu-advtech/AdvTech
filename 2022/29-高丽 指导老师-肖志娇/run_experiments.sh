#!/usr/local/bin/bash
#
# Requires Bash v4 for associative arrays.
# Mac install with Homebrew will make it available under /usr/local/bin.

# Globals.
TODAY=`date "+%Y%m%d"`
NUMBER_RUNS=10
Y_AXIS_COLUMN=runtime_seconds  # You probably do not want to change this.

# Experiment -> dataset directory
declare -A EXPERIMENT_DATASETS
EXPERIMENT_DATASETS["column_length"]=synthetic_datasets/fixed_row_size/structured_text_currency_codes
EXPERIMENT_DATASETS["avg_row_length"]=synthetic_datasets/fixed_col_size/different_hinges_and_free_form_text
EXPERIMENT_DATASETS["avg_num_hinges"]=synthetic_datasets/fixed_col_size/structured_text_same_hinges
EXPERIMENT_DATASETS["max_num_hinges"]=synthetic_datasets/fixed_col_size/structured_text_same_hinges
EXPERIMENT_DATASETS["num_branches"]=synthetic_datasets/fixed_col_size/different_hinges_and_free_form_text-single_column

# Run experiment and generate logs.
function run_experiment {
  local input_dataset_dir=$1
  local output_log_file=$2
  local number_runs=$3

  args="-m timeit -n 1 -r ${number_runs} 'import os' 'os.system(\"python main.py -d ${input_dataset_dir}\")' > ${output_log_file} 2>&1"
  echo "Running:"
  echo "python $args"
  eval python $args
  echo "Saved logs as \"${output_log_file}\"."
}

# Generate CSV from experiment logs.
function generate_csv {
  local input_log_file=$1
  local output_csv_file=$2

  # Generate CSV from experiment logs.
  echo "Generating CSV..."
  grep learn_structures_for_single_column ${input_log_file} | sed -e $'1i\\\nfunction,runtime_seconds,column_length,avg_row_length,avg_num_hinges,max_num_hinges,num_branches' > ${output_csv_file}
  echo "Saved CSV as \"${output_csv_file}\"."
}

# Generates plot from CSV.
function plot_csv {
  local input_csv_file=$1
  local output_plot_file=$2
  local x_axis_column=$3
  local y_axis_column=$4

  echo "Plotting results..."
  python plot.py -i ${input_csv_file} -o ${output_plot_file} -x ${x_axis_column} -y ${y_axis_column}
  echo "Saved plot as \"${output_plot_file}\"."
}

for experiment in "${!EXPERIMENT_DATASETS[@]}";
do
  x_axis_column=$experiment
  input_dataset_dir=${EXPERIMENT_DATASETS[$experiment]}
  file_basename=${TODAY}-${experiment}
  log_filename=logs/${file_basename}.log
  csv_filename=logs/${file_basename}.csv
  plot_filename=plots/${file_basename}.png

  run_experiment $input_dataset_dir $log_filename $NUMBER_RUNS
  generate_csv $log_filename $csv_filename
  plot_csv $csv_filename $plot_filename $x_axis_column $Y_AXIS_COLUMN
done

#  使用read命令达到类似bat中的pause命令效果
echo 按任意键继续
read -n 1
echo 继续运行
