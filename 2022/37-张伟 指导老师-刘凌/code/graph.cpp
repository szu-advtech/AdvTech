#include "graph.h"
 
// SeedIO
std::vector<int> SeedIO::Read(const std::string& filename, IGraph& gf)
{
	std::ifstream ft(filename.c_str());
	return Read(ft, gf);
}

std::vector<int> SeedIO::Read(std::istream& in, IGraph& gf)
{
	std::string ss;
	std::vector<int> seeds;
	int nums = 0;
	int seedCount = 0;

	bool isNumFinished = false;

	// read seeds
	while (getline(in, ss))
	{
		// skip empty line or comment line
		if (isEmptyOrCommentLine(ss)) {
			continue;
		}

		// read the seed number (just once)
		if (!isNumFinished) {
			std::stringstream ssinout(ss);
			ssinout >> nums;   // # of seeds
			assert(nums > 0);
			isNumFinished = true;
			continue;
		}

		// read one seed
		if (seedCount < nums) {
			std::stringstream ssinout(ss);
			std::string seedStr;
			ssinout >> seedStr;
			int s = gf.MapNodeNameToIndex(seedStr);
			seeds.push_back(s);
			seedCount++;
		}
		else {
			break;
		}
	}

	if (seedCount != nums) {
		std::string ss = "Seed input is incorrect! Expect #seeds = ";
		ss += std::to_string(nums);
		ss += ", find #seeds = ";
		ss += std::to_string(seedCount);
		throw InvalidInputFormatException(ss);
	}
	return seeds;
}


std::vector<int> SeedIO::Read(const std::string& filename, std::vector<double>& out_infl, IGraph& gf)
{
	std::ifstream ft(filename.c_str());
	return Read(ft, out_infl, gf);
}

std::vector<int> SeedIO::Read(std::istream& in, std::vector<double>& out_infl, IGraph& gf)
{
	std::string ss;
	std::vector<int> seeds;
	out_infl.clear(); // clear the out_infl
	int nums = 0;
	int seedCount = 0;

	bool isNumFinished = false;
	out_infl.clear();

	// read seeds
	while (getline(in, ss))
	{
		// skip the comment line
		if (ss[0] == '#') {
			continue;
		}

		// read the seed number (just once)
		if (!isNumFinished) {
			std::stringstream ssinout(ss);
			ssinout >> nums;   // # of seeds
			assert(nums > 0);
			isNumFinished = true;
			continue;
		}

		// read one seed
		if (seedCount < nums) {
			std::stringstream ssinout(ss);
			std::string seedStr;
			ssinout >> seedStr;
			int s = gf.MapNodeNameToIndex(seedStr);
			double d = -1;
			ssinout >> d;
			seeds.push_back(s);
			out_infl.push_back(d);
			seedCount++;
		}
		else {
			break;
		}
	}

	if (seedCount != nums) {
		std::string ss = "Seed input is incorrect! Expect #seeds = ";
		ss += std::to_string(nums);
		ss += ", find #seeds = ";
		ss += std::to_string(seedCount);
		throw InvalidInputFormatException(ss);
	}
	return seeds;
}


void SeedIO::Write(const std::string& filename,
	const std::vector<int>& seeds,
	const std::vector<double>& infl, IGraph& gf)
{
	std::ofstream ft(filename.c_str());
	Write(ft, seeds, infl, gf);
}

void SeedIO::Write(std::ostream& out,
	const std::vector<int>& seeds,
	const std::vector<double>& infl, IGraph& gf)
{
	size_t num = seeds.size();
	out << num << std::endl;
	for (size_t i = 0; i < num; i++) {
		out << gf.MapIndexToNodeName(seeds[i]) << "\t" << infl[i] << std::endl;
	}
}


