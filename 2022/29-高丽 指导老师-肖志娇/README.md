#Current Status
Rigid structure is being learned through an altered dafsa, with ability to add words and (to some extent) detect structure.

#Usage

```
$ python main.py --help
usage: main.py [-h] (-f FILENAME | -d DIRECTORY)

Learn structures model for a single file, or all files in a directory.

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Relative path of input filename to be processed.
  -d DIRECTORY, --directory DIRECTORY
                        Relative path of input directory containing all files
                        to be processed.
```

#Generating Data
```
$ python data_generator.py --help
usage: data_generator.py [-h] -f FILENAME -n NUMROWS -t NUMTYPES

Generates a single-column text file with sample data obeying specified
parameters

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Relative path of the output file
  -n NUMROWS, --numrows NUMROWS
                        Number of rows to generate
  -t NUMTYPES, --numtypes NUMTYPES
                        Number of datatypes to mix
```
#Running Micro-benchmark Experiments

From the root directory:

```
$ ./run_experiments.sh
```

# TODO
- [x] Detecting the presence of structure through modified edit distance
- [x] Building a DAFSA-like structure for learning regular expressions 
- [x] DAFSA layer-collapsing 
- [x] Learning column structure end-to-end 
- [x] Inclusion of MIT dataset for testing 
- [x] Early-convergence stop condition 
- [ ] Regex comparisons 
- [ ] Testing on data.gov 
- [ ] Benchmarking 
- [x] __Hinge discovery__  
