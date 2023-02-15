
#pragma once
#include <windows.h>
#include <Commdlg.h>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <omp.h>
#include <string>
#include <queue>

#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include<gui/shader.h>
#include<gui/camera.h>
#include<gui/File_M.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <CGAL/memory.h>
#include <CGAL/IO/Color.h>
#include <CGAL/Iterator_range.h>
#include <CGAL/HalfedgeDS_vector.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing_on_polygon_mesh.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>

#include <filesystem>

#include <Eigen/Core>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>

namespace fs = std::filesystem;
namespace PS = CGAL::Polyline_simplification_2;
typedef PS::Stop_below_count_ratio_threshold Stop;
typedef PS::Squared_distance_cost            Cost;
// Type declarations.

//struct K : CGAL::Filtered_kernel_adaptor<CGAL::Type_equality_wrapper<CGAL::Simple_cartesian<mpq_class>::Base<K>::Type, K>, true> {};
typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Exact_predicates_exact_constructions_kernel IK;
typedef CGAL::Exact_predicates_inexact_constructions_kernel inexact_K;

typedef CGAL::Polyhedron_3<IK> Polyhedron;
typedef CGAL::Nef_polyhedron_3<IK> Nef_polyhedron;


typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef CGAL::Surface_mesh<IK::Point_3> IK_Mesh;
typedef Mesh::Vertex_index it_V;
typedef Mesh::Edge_index it_E;
typedef Mesh::Face_index it_F;
typedef glm::vec3 Vec3;
typedef glm::vec2 Vec2;
typedef std::pair<int, int>PII;
typedef K::Point_3 Point_3;
typedef K::Point_2 Point_2;
typedef K::Plane_3 plane_3;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Polygon_2<IK> IK_Polygon_2;
typedef CGAL::Polygon_set_2<K> Polygon_set_2;

namespace PMP = CGAL::Polygon_mesh_processing;
using FT = typename K::FT;
using Color = CGAL::IO::Color;
// Choose the type of a container for a polygon mesh.
using EK_to_IK = CGAL::Cartesian_converter<K, IK>;
using IK_to_EK = CGAL::Cartesian_converter<IK, K>;
using IK_to_IEK = CGAL::Cartesian_converter<IK, inexact_K>; 
using EK_to_IEK = CGAL::Cartesian_converter<K, inexact_K>;
using IEK_to_IK = CGAL::Cartesian_converter<inexact_K,IK>;
using Polygon_mesh = CGAL::Surface_mesh<Point_3>;
using Face_range = typename Polygon_mesh::Face_range;
using Neighbor_query = CGAL::Shape_detection::Polygon_mesh::One_ring_neighbor_query<Polygon_mesh>;
using Region_type = CGAL::Shape_detection::Polygon_mesh::Least_squares_plane_fit_region<K, Polygon_mesh>;
using Sorting = CGAL::Shape_detection::Polygon_mesh::Least_squares_plane_fit_sorting<K, Polygon_mesh, Neighbor_query>;

using Region = std::vector<std::size_t>;
using Regions = std::vector<Region>;
using Vertex_to_point_map = typename Region_type::Vertex_to_point_map;
using Region_growing = CGAL::Shape_detection::Region_growing<Face_range, Neighbor_query, Region_type, typename Sorting::Seed_map>;


typedef CGAL::Polygon_with_holes_2<IK>                Polygon_with_holes_2;
typedef std::vector<Polygon_with_holes_2>                   Pwh_vec_2;

typedef boost::graph_traits<IK_Mesh>::vertex_descriptor vertex_descriptor;
typedef IK_Mesh::Property_map<vertex_descriptor, K::Point_3> Exact_point_map;
typedef Eigen::Matrix<IK::FT, Eigen::Dynamic, 3> MatrixX3E;
typedef std::pair<int, int>PII;


namespace params = PMP::parameters;
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

struct Exact_vertex_point_map
{
	// typedef for the property map
	typedef boost::property_traits<Exact_point_map>::value_type value_type;
	typedef boost::property_traits<Exact_point_map>::reference reference;
	typedef boost::property_traits<Exact_point_map>::key_type key_type;
	typedef boost::read_write_property_map_tag category;
	// exterior references
	Exact_point_map exact_point_map;
	IK_Mesh* tm_ptr;
	// Converters
	CGAL::Cartesian_converter<IK, K> to_exact;
	CGAL::Cartesian_converter<K, IK> to_input;
	Exact_vertex_point_map()
		: tm_ptr(nullptr)
	{}
	Exact_vertex_point_map(const Exact_point_map& ep, IK_Mesh& tm)
		: exact_point_map(ep)
		, tm_ptr(&tm)
	{
		for (Mesh::Vertex_index v : vertices(tm))
			exact_point_map[v] = to_exact(tm.point(v));
	}
	friend
		reference get(const Exact_vertex_point_map& map, key_type k)
	{
		CGAL_precondition(map.tm_ptr != nullptr);
		return map.exact_point_map[k];
	}
	friend
		void put(const Exact_vertex_point_map& map, key_type k, const K::Point_3& p)
	{
		CGAL_precondition(map.tm_ptr != nullptr);
		map.exact_point_map[k] = p;
		// create the input point from the exact one
		map.tm_ptr->point(k) = map.to_input(p);
	}
};

struct mesh_for_Render {
	std::vector<Vec3> tri_V_C_N;
	std::vector<Vec3> point_V_C_N;
	std::vector<Vec3> line_V_C_N;
	std::vector<Vec3> V_Idx;

	std::vector<int> idxs;
};
struct simplex {//q:can we just use simplex to express mesh/obj
	int id, level;
	simplex() {
		id = 0, level = 0;
	}
	simplex(int a, int b) {
		id = a, level = b;
	}
	void operator= (const struct simplex& other) {
		id = other.id;
		level = other.level;
	}
};
struct Regions_plane
{
	std::vector<it_F> face_its;
	std::unordered_set<it_F>fds_set;
	plane_3 plane;
	double area;
	bool operator<(const Regions_plane& other)const
	{
		return area > other.area;
	}
};

struct Visual_hull_view
{
	Visual_hull_view(plane_3& plane_, Polygon_with_holes_2& pwh_) :
		plane(plane_), pwh(pwh_)
	{ }
	plane_3 plane;
	Polygon_with_holes_2 pwh;
};


struct normal_area
{
	double area=0.0;
	CGAL::Vector_3<K> normal;
};
struct view_direction
{
	double area=0.0;
	CGAL::Vector_3<K> Dir;
	bool operator<(const view_direction& other)const
	{
		return area > other.area;
	}
};

class Mergeset
{
public:
	int p[100000];
	void init(){
		for (int i = 0; i < 100000; ++i)p[i] = -1;
	}
	Mergeset() {
		for (int i = 0; i < 100000; ++i)p[i] = -1;
	}
	int find(int x)
	{
		if (p[x] != x)
			p[x] = find(p[x]);
		return p[x];
	}
	void merge(int x, int y)
	{
		p[y] = find(p[x]);
	}
};

class plane_Mergeset
{
public:
	int p[10000];
	CGAL::Vector_3<K> normals[10000];
	plane_Mergeset() {
		for (int i = 0; i < 10000; ++i)p[i] = i;
	}
	int find(int x)
	{
		return p[x] == x ? x : p[x] = find(p[x]);
	}
	void merge(int x, int y)
	{
		p[y] = find(p[x]);
	}
};

struct libgl_mesh
{
	MatrixX3E V;
	Eigen::MatrixXi F;
};

class Mesh_area
{
public:
	Mesh_area(const IK_Mesh& _mesh,const double& _area) :
		mesh(_mesh), area(_area) { }
	IK_Mesh mesh;
	double area;
	bool operator<(const Mesh_area& other)const
	{
		return area > other.area;
	}
};

class Runner {
public:
	Runner(){};
	std::string save_path="";
	int run(Runner& runner);
	bool open_mesh();
	bool save_mesh(const fs::path& path);
	Mesh mesh;
	IK_Mesh ik_mesh,Visual_hull_mesh,output_mesh;

	Nef_polyhedron Visual_hull,Carved_Mesh;
	std::vector<std::unique_ptr<IK_Mesh>> mesh_Vs;
	std::vector<std::unique_ptr<IK_Mesh>> primitives;
	std::vector<Nef_polyhedron> primitives_polyhedron;
	std::vector<std::unique_ptr<IK_Mesh>> render_primitives;
	std::vector<std::unique_ptr<IK_Mesh>> carve_meshes;
	std::vector<Nef_polyhedron> Nef_carve_meshes;
	std::vector<Regions_plane> carve_planes;
	std::vector<std::unordered_set<int>> faces_can_see;

	mesh_for_Render render_mesh,render_mesh_V;
	unsigned int tri_VBO, tri_VAO, tri_EBO,
		map_VAO,map_VBO,map_EBO,
		line_VAO,line_VBO,line_EBO,
		point_VAO,point_VBO,point_EBO,
		mtri_VBO, mtri_VAO, mtri_EBO,
		mline_VAO, mline_VBO, mline_EBO,
		mpoint_VAO, mpoint_VBO, mpoint_EBO,View_buffer;

	Shader shader,Primitives_shader,map_shader;
	//i need a method to discriminate this with surface mesh p_id ! p_id means surface mesh id
	//a method to transform between this,maybe map,lots of idxs to one p_id
	void select(int xpos, int ypos);
	Vec2 mvp_transform(Vec3 point);
	float distance(Vec2 p1, Vec2 p2, Vec2 p3);
	float distance(Vec2 p1, Vec2 p2);
	glm::mat4 m_model, m_view, m_projection;
	std::vector<CGAL::SM_Face_index>selected_face;
	std::vector<CGAL::SM_Edge_index>selected_edge;
	std::vector<CGAL::SM_Vertex_index>selected_vertex;

	std::vector<PII> E_V_matrix;
	std::vector<PII> F_E_matrix;
	std::vector<bool> vertices_vec;
	std::vector<bool> edges_vec;
	std::vector<bool> faces_vec;
	std::vector<simplex> input_;
	std::vector<simplex> result_;

	std::vector<Regions_plane> regions_plane;
	std::vector<normal_area> normal_areas;
	std::vector<view_direction> All_views;
	std::vector<view_direction> after_merged_All_views;
	std::vector<view_direction>Visual_hull_views;
	std::vector<view_direction>measure_views;
	std::vector<Visual_hull_view> V_hull_views;
	std::unordered_set<CGAL::SM_Face_index> detected_fds;

	std::unique_ptr<IK_Mesh> to_surface_mesh(libgl_mesh & mesh);
	std::unique_ptr<libgl_mesh> to_libigl_mesh(IK_Mesh & mesh);
	std::unique_ptr<Polyhedron> to_Polyhedron(IK_Mesh& mesh);

	void get_matrices();
	void build_bool_vec();
	void get_input();
	void clear_result();

	void generate_Visual_hull_views();
	void detect_planes();//this is very matter
	void init_origin_measure(Shader& shader);
	void render_for_primitives();
	//double difference_metric(Mesh& new_mesh,Shader& shader,bool use_boolean_type);
	double difference_metric(IK_Mesh& new_mesh, Shader& shader, bool use_boolean_type);
	double difference_metric(libgl_mesh& new_mesh, Shader& shader, bool use_boolean_type);
	double difference_metric_polyhedron(Polyhedron& new_mesh, Shader& shader, bool use_boolean_type);



	void set_M_v_bbox();
	void get_primitives();
	void get_carved_meshes();
	bool mesh_to_polyhedron(const IK_Mesh& mesh, Polyhedron& polyhedron);
	
	void compute_M_v();// todo:accelerate
	void carving();

	bool my_boolean_op(IK_Mesh& mesh1, IK_Mesh& mesh2, std::string op);


	IK::Point_3 get_P(unsigned int f,int x,int y);
	Polygon_set_2 divide_conquer_join(std::vector<Polygon_2> polygons,int l,int r);
	IK::Plane_3 get_unit_plane(IK::Plane_3 plane);	
	bool got_render_mesh = false;
	void get_mesh(std::vector<Point_3>& points, std::vector<std::vector<unsigned int>> indices);
	void handle_result_();
	void star();
	void closure();
	void link();
	void select_reset();
	bool is_sim_complex();
	int is_pure_complex();
	void boundary();
	bool _sim_complex = false;
	int k_pure_complex=-1;
	static void cur_call_back(GLFWwindow* window, double xpos, double ypos);
	static void frame_buffer_callback(GLFWwindow* window, int w, int h);
	static void my_ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

	void render_obj(Shader& shader);
	void render_map(Shader& shader);
	std::string window_select_file();
	bool read_mesh(std::string path);
	//void transform(Mesh& mesh,mesh_for_Render& render_mesh);
	void transform_ik(IK_Mesh& mesh, mesh_for_Render& render_mesh);
	void transform(libgl_mesh& mesh, mesh_for_Render& render_mesh);
	void transform_polyhedron(Polyhedron& mesh, mesh_for_Render& render_mesh);
	Vec3 transform_idx(unsigned int idx);
	unsigned int reverse_transform_idx(Vec3 idx_code);
	Vec3 tri_inited_color = Vec3(0.78,0.78,0.98);
	Vec3 point_inited_color = Vec3(0, 0, 0);
	Vec3 line_inited_color = Vec3(0, 0, 0);
	Vec3 selected_point_color = Vec3(0.8, 0.4, 0.1);
	Vec3 selected_line_color = Vec3(0.8, 0.4, 0.1);
	Vec3 selected_tri_color = Vec3(0.8, 0.4, 0.1);
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


	int dir_id = 0;
	private:
};
