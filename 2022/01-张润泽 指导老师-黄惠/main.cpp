#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include <getopt/getopt.hpp>
#include "process.h"
#include "config.h"

int main()
{
	const std::string config_file = getarg("config.toml", "--config");
	try
	{
		Config::tbl = toml::parse_file(config_file);
		std::cout << Config::tbl << "\n" << std::endl;
	}
	catch (const toml::parse_error& err)
	{
		std::cerr << config_file << " Parsing failed:\n" << err << "\n";
		return 1;
	}
	if (Config::Path::get().save_path == "") {
		std::cout << "set save path in config\n";
		return 1;
	}
	Runner runner;
	runner.save_path = Config::Path::get().save_path;
	runner.run(runner);
}