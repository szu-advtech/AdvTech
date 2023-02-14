#pragma once

#define TOML_HEADER_ONLY 0
#include <toml++/toml.h>
#include <string>
#include <iostream>
#include <fmt/core.h>

namespace Config {
	extern toml::table tbl;

#define PARAM_GROUP(CLASS_Name, GROUP)													\
	private:																			\
		CLASS_Name() : group(GROUP), nv(tbl[GROUP]) {};									\
		CLASS_Name(const CLASS_Name&) = delete;											\
		CLASS_Name& operator=(const CLASS_Name&) = delete;								\
		template<typename T>															\
		T read(const std::string& key) const {											\
			auto value_opt = nv[key].value<T>();										\
			if (!value_opt) {															\
				std::cout << group << " :error getting config " << key << std::endl;	\
				std::exit(1);															\
			}																			\
			return *value_opt;															\
		}																				\
		static inline CLASS_Name* data;													\
		std::string group;																\
		toml::node_view<toml::node> nv;													\
	public:																				\
		static CLASS_Name& get() {														\
			if (!data) data = new CLASS_Name();											\
				return *data;															\
		}																			


// init config value from the [toml] config file
#define DECLA(TYPE, NAME)			\
	TYPE NAME = read<TYPE>(#NAME)		

	class Path
	{
		PARAM_GROUP(Path, "path");
	public:
		DECLA(std::string, save_path);
		DECLA(int, planes);
		DECLA(int, views);
	};

#undef PARAM_GROUP
#undef DECLA

	template<typename T>
	T read(const std::string &key) {
		auto value_opt = tbl[key].value<T>();
		if (!value_opt) {
			std::cout << "error getting config " << key << std::endl;
			std::exit(1);
		}
		return *value_opt;
	}


}
