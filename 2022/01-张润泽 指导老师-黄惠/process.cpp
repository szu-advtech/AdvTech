#include"process.h"
#include "config.h"
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
// Midpoint placement policy
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>
//Placement wrapper
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Constrained_placement.h>
// Stop-condition policy
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>

template<class BasePlacement, class EdgeIsConstrainedMap>
class my_Constrained_placement
    : public BasePlacement
{
public:
    my_Constrained_placement(const EdgeIsConstrainedMap map = EdgeIsConstrainedMap(),
        const BasePlacement& base = BasePlacement())
        : BasePlacement(base),
        m_ecm(map)
    {}

    template <typename Profile>
    boost::optional<typename Profile::Point> operator()(const Profile& profile) const
    {
        typedef typename Profile::TM                                    TM;
        typedef typename boost::graph_traits<TM>::halfedge_descriptor   halfedge_descriptor;

        for (halfedge_descriptor h : halfedges_around_target(profile.v0(), profile.surface_mesh()))
        {
            if (get(m_ecm, edge(h, profile.surface_mesh())))
            {
                CGAL::SM_Vertex_index index = profile.v0();
                IK_Mesh::Property_map<CGAL::SM_Vertex_index,bool> &map = profile.surface_mesh().property_map<CGAL::SM_Vertex_index, bool>
                    ("is_deleted").first;
                if (!map[index])
                {
                    return get(profile.vertex_point_map(), profile.v0());
                }
            }
        }

        for (halfedge_descriptor h : halfedges_around_target(profile.v1(), profile.surface_mesh()))
        {
            if (get(m_ecm, edge(h, profile.surface_mesh())))
            {
                CGAL::SM_Vertex_index index = profile.v1();
                IK_Mesh::Property_map<CGAL::SM_Vertex_index, bool>& map = profile.surface_mesh().property_map<CGAL::SM_Vertex_index, bool>
                    ("is_deleted").first;
                if (!map[index])
                    return get(profile.vertex_point_map(), profile.v1());
            }
        }

        return static_cast<const BasePlacement*>(this)->operator()(profile);
    }

private:
    EdgeIsConstrainedMap m_ecm;
};


template<class TM_>
class my_point_placement
{
public:
    typedef TM_                                                     TM;

    my_point_placement() {}

    template <typename Profile>
    boost::optional<typename Profile::Point> operator()(const Profile& profile) const
    {
        typedef boost::optional<typename Profile::Point>              result_type;
        
        IK_Mesh::Property_map<CGAL::SM_Vertex_index, bool>& map = profile.surface_mesh().property_map<CGAL::SM_Vertex_index, bool>
            ("is_deleted").first;
        CGAL::SM_Vertex_index index0 = profile.v0();
        if (!map[index0])
        {
            return result_type(get(profile.vertex_point_map(), profile.v0()));
        }
        else return result_type(get(profile.vertex_point_map(), profile.v1()));
    }
};

typedef boost::graph_traits<IK_Mesh>::halfedge_descriptor  halfedge_descriptor;
typedef boost::graph_traits<IK_Mesh>::edge_descriptor      edge_descriptor;
namespace SMS = CGAL::Surface_mesh_simplification;
// BGL property map which indicates whether an edge is marked as non-removable
struct Border_is_constrained_edge_map
{
    const IK_Mesh* sm_ptr;
    typedef edge_descriptor                                       key_type;
    typedef bool                                                  value_type;
    typedef value_type                                            reference;
    typedef boost::readable_property_map_tag                      category;
    Border_is_constrained_edge_map(const IK_Mesh& sm) : sm_ptr(&sm) {}
    friend value_type get(const Border_is_constrained_edge_map& m, const key_type& edge) {
        return !(*m.sm_ptr).property_map<CGAL::SM_Edge_index, bool>
            ("is_deleted").first[edge];
    }
};
// Placement class
typedef SMS::Constrained_placement<my_point_placement<IK_Mesh>,
    Border_is_constrained_edge_map > Placement;
//--------------------------------------------------------------
std::string SAVE_PATH;
Camera camera = Camera(glm::vec3(0,0,3));
float fov = 45.0f;
int SCR_WIDTH = 1280, SCR_HEIGHT = 720;
bool on_select = false;
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;
unsigned int framebuffer;
unsigned int measure_frame_buffer[1000];
unsigned int measure_framebuffer_new_mesh;
Runner *instance;
float transform_map[256];
GLFWwindow* window;
std::unordered_map<int, int> pidx_to_pid;
static int select_simplex_level;
float matrix[128 * 128 * 3] = {0.0};
float pos_matrix[128 * 128 * 3] = { 0.0 };
float View_matrix[1280][720][3] = { 0.0 };
bool found[128][128] = {0};
Eigen::MatrixXd VA, VB, VC;
Eigen::VectorXi J, I;
Eigen::MatrixXi FA, FB, FC;
int measure_w = 500, measure_h = 500;
float m_i[500 * 500 * 3] = { 0.0 };
float m_v[500 * 500 * 3] = { 0.0 };
Mergeset m;
int dx[] = { 0,0,1,1,1,-1,-1,-1 };
int dy[] = { 1,-1,0,1,-1,0,1,-1 };
IK::Point_3 NULL_POINT(1121, 222, 33333);
std::vector<std::vector<unsigned int>> face_link_matrix;
plane_Mergeset ms;
plane_Mergeset view_set;
double areas[10000] = { 0.0 };
Runner* my_runner;

struct FaceInfo2
{
    FaceInfo2() {}
    int nesting_level;
    bool in_domain() {
        return nesting_level % 2 == 1;
    }
};

typedef CGAL::Triangulation_vertex_base_2<IK>                      Vb;
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, IK>    Fbb;
typedef CGAL::Constrained_triangulation_face_base_2<IK, Fbb>        Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>               TDS;
typedef CGAL::Exact_intersections_tag                                Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<IK, TDS, Itag>  CDT;
typedef CGAL::Constrained_triangulation_plus_2<CDT>                       CDTP;
typedef CDT::Point                                                CDT_Point;
typedef CGAL::Polygon_2<IK>                                        CDT_Polygon_2;
typedef CDTP::Face_handle                                          Face_handle;
void
mark_domains(CDTP& ct,
    Face_handle start,
    int index,
    std::list<CDTP::Edge>& border)
{
    if (start->info().nesting_level != -1) {
        return;
    }
    std::list<Face_handle> queue;
    queue.push_back(start);
    while (!queue.empty()) {
        Face_handle fh = queue.front();
        queue.pop_front();
        if (fh->info().nesting_level == -1) {
            fh->info().nesting_level = index;
            for (int i = 0; i < 3; i++) {
                CDTP::Edge e(fh, i);
                Face_handle n = fh->neighbor(i);
                if (n->info().nesting_level == -1) {
                    if (ct.is_constrained(e)) border.push_back(e);
                    else queue.push_back(n);
                }
            }
        }
    }
}
//explore set of facets connected with non constrained edges,
//and attribute to each such set a nesting level.
//We start from facets incident to the infinite vertex, with a nesting
//level of 0. Then we recursively consider the non-explored facets incident
//to constrained edges bounding the former set and increase the nesting level by 1.
//Facets in the domain are those with an odd nesting level.
void
mark_domains(CDTP& cdt)
{
    for (CDTP::Face_handle f : cdt.all_face_handles()) {
        f->info().nesting_level = -1;
    }
    std::list<CDTP::Edge> border;
    mark_domains(cdt, cdt.infinite_face(), 0, border);
    while (!border.empty()) {
        CDTP::Edge e = border.front();
        border.pop_front();
        Face_handle n = e.first->neighbor(e.second);
        if (n->info().nesting_level == -1) {
            mark_domains(cdt, n, e.first->info().nesting_level + 1, border);
        }
    }
}
void processInput(GLFWwindow* window);
bool simplify(IK_Mesh& surface);
void show()
{
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);//here because it's 0
        {
            glClearColor(my_runner->clear_color.x, my_runner->clear_color.y, 
                my_runner->clear_color.z, my_runner->clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
        ImGui::Begin("...");
        {
            ImGui::InputInt("input select_simplex_level", &select_simplex_level);
            if (ImGui::Button("open")) {//todo open & clear 
                if (!my_runner->open_mesh()) {
                    return;
                }
                my_runner->render_obj(my_runner->shader);
                my_runner->render_map(my_runner->map_shader);
                my_runner->got_render_mesh = true;
                my_runner->detect_planes();
                my_runner->init_origin_measure(my_runner->shader);
                ////get_matrices();
                my_runner->set_M_v_bbox();
                my_runner->get_carved_meshes();
                for (int i = 0; i < my_runner->carve_meshes.size(); ++i)
                {
                    const std::string path = SAVE_PATH;
                    const std::string fullpath = path + "carve_mesh" + std::to_string(i) + ".off";
                    std::ofstream out(fullpath);
                    out << (*my_runner->carve_meshes[i]);
                    Polyhedron p;
                    std::ifstream ifs1(fullpath);
                    ifs1 >> p;
                    Nef_polyhedron N_p(p);
                    if (N_p.is_simple())
                        my_runner->Nef_carve_meshes.emplace_back(N_p);
                }
                auto start = clock();
                my_runner->get_primitives();
                auto end = clock();
                double endtime = (double)(end - start) / CLOCKS_PER_SEC;
                std::cout << "Total time:" << endtime << '\n';

                std::cout << "primitives_size:" << my_runner->primitives.size() << '\n';
                for (int i = 0; i < my_runner->primitives.size(); ++i)
                {
                    const std::string path = SAVE_PATH;
                    const std::string fullpath = path + std::to_string(i) + ".off";
                    std::ofstream out(fullpath);
                    out << (*my_runner->primitives[i]);
                    Polyhedron p;
                    std::ifstream ifs1(fullpath);
                    ifs1 >> p;
                    Nef_polyhedron N_p(p);
                    my_runner->primitives_polyhedron.emplace_back(N_p);
                }
                my_runner->compute_M_v();

                const std::string path = SAVE_PATH;
                const std::string fullpath = path + "visual_hull_end" + ".off";
                std::ofstream out(fullpath);
                out << (*my_runner->mesh_Vs.back());
                out.close();

                IK_Mesh output;
                CGAL::convert_nef_polyhedron_to_polygon_mesh(my_runner->Visual_hull, output);
                const std::string fullpath_P = path + "visual_hull_end_polyhedra" + ".off";
                std::ofstream out_P(fullpath_P);
                out_P << output;
                out_P.close();

                my_runner->carving();

                const std::string fullpath_ = path + "carve_end" + ".off";
                std::ofstream out_(fullpath_);
                out_ << (*my_runner->mesh_Vs.back());
                out.close();
                /*     transform(*mesh_Vs.back(), render_mesh);*/
            }
            ImGui::SameLine();
            if (ImGui::Button("show_next_primitives")) {
                //transform(*primitives[pri_id], render_mesh);

            }
            ImGui::SameLine();
            //if (ImGui::Button("select")){
            //    on_select = !on_select;//cursor callback use this bool
            //}
            //ImGui::SameLine();
            if (ImGui::Button("check_out")) {
                my_runner->transform_ik(*my_runner->mesh_Vs.back(), my_runner->render_mesh);
            }
            ImGui::SameLine();
            if (ImGui::Button("input")) {
                my_runner->transform_ik(my_runner->mesh, my_runner->render_mesh);
            }
            ImGui::SameLine();
            //if (ImGui::Button("finish_select")) {
            //    select_reset();
            //}
            ImGui::SameLine();
            if (ImGui::Button("save")) {
                //save_result();
            }
            ImGui::SameLine();
            if (ImGui::Button("simplify an out put")) {
                simplify(my_runner->ik_mesh);
                const std::string path = SAVE_PATH;
                const std::string fullpath = path + "simplfied" + ".off";
                std::ofstream out(fullpath);
                out << (my_runner->ik_mesh);
            }
            ImGui::SameLine();
            if (ImGui::Button("check view_dir")) {
                auto temp_dir = my_runner->measure_views[my_runner->dir_id++].Dir;
                std::cout << temp_dir << '\n';
                auto unit_temp_dir = (Vec3(
                    CGAL::to_double(temp_dir.x())
                    , CGAL::to_double(temp_dir.y())
                    , CGAL::to_double(temp_dir.z())));
                auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;
                glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
                glm::mat4 projection = glm::ortho(-1.5f, 1.5f, -1.5f, 1.5f, -1.0f, 100.0f);
                glm::mat4 view = glm::lookAt(temp_eye, Vec3(0.f, 0.f, 0.f), Vec3(0.0f, 1.0f, 0.0f));
                glm::mat4 model = glm::mat4(1);
                my_runner->m_model = model, my_runner->m_view = view, my_runner->m_projection = projection;
            }
            if (ImGui::TreeNode("Output"))
            {
                ImGui::Text("is_sim_complex: %d", my_runner->_sim_complex);
                ImGui::Text("k_sim_complex: %d", my_runner->k_pure_complex);
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Operators"))
            {
                ImGui::Bullet();
                if (ImGui::SmallButton("closure")) {
                    my_runner->closure();
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("star")) {
                    my_runner->star();
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("link")) {
                    my_runner->link();
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("is_sim_complex")) {
                    if (my_runner->is_sim_complex()) {
                        my_runner->_sim_complex = 1;
                    }
                    else my_runner->_sim_complex = 0;
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("is_pure_sim_complex")) {
                    my_runner->k_pure_complex = my_runner->is_pure_complex();
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("boundary")) {
                    my_runner->boundary();
                }
                ImGui::TreePop();
            }
        }
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        //------------------render_obj----------------------------------------------------------

        if (my_runner->got_render_mesh)
        {
            //render_tris
            {//show_primitives
                /*auto temp_dir = Visual_hull_views[pri_id].Dir;
                auto unit_temp_dir = Vec3(
                    CGAL::to_double(temp_dir.x()) / std::sqrt(CGAL::to_double(temp_dir.squared_length()))
                    , CGAL::to_double(temp_dir.y()) / std::sqrt(CGAL::to_double(temp_dir.squared_length()))
                    , CGAL::to_double(temp_dir.z()) / std::sqrt(CGAL::to_double(temp_dir.squared_length())));

                auto temp_eye = Vec3(0, 0, 0) - Vec3(
                    (double)3.0 / CGAL::to_double(temp_dir[2]) * unit_temp_dir.x,
                    (double)3.0 / CGAL::to_double(temp_dir[2]) * unit_temp_dir.y,
                    (double)3.0 / CGAL::to_double(temp_dir[2]) * unit_temp_dir.z);*/
                    //glm::mat4 view = glm::lookAt(temp_eye, unit_temp_dir, camera.up());
            }

            my_runner->render_obj(my_runner->shader);
            my_runner->shader.use();
            my_runner->shader.setInteger("u", 1);
            glEnable(GL_DEPTH_TEST);
            //glm::mat4 projection = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.1f, 100.0f);
            glm::mat4 projection = glm::perspective(glm::radians(fov), (float)1280 / (float)720, 0.1f, 100.0f);
            glm::mat4 view = glm::lookAt(camera.eye(), camera.dir(), camera.up());

            glm::mat4 model = glm::mat4(1);

            my_runner->shader.setMat4("model", model);
            my_runner->shader.setMat4("view", view);
            my_runner->shader.setMat4("projection", projection);
            glBindVertexArray(my_runner->tri_VAO);
            glDrawElements(GL_TRIANGLES, my_runner->render_mesh.idxs.size(), GL_UNSIGNED_INT, 0);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(0);

            //render lines
// if different color for point,edge,tris,use three different vao,vbo;

            glEnable(GL_DEPTH_TEST);
            glEnable(GL_LINE_SMOOTH);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glLineWidth(1.5f);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            //glBindVertexArray(line_VAO);
            //glDrawElements(GL_TRIANGLES, (GLuint)render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBindVertexArray(my_runner->line_VAO);
            glDrawArrays(GL_LINES, 0, my_runner->render_mesh.line_V_C_N.size() / 3);
            glDisable(GL_LINE_SMOOTH);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glBindVertexArray(0);

            // render points

            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glPointSize(1.0);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            //glBindVertexArray(point_VAO);
            //glDrawElements(GL_TRIANGLES, render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBindVertexArray(my_runner->point_VAO);
            glDrawArrays(GL_POINTS, 0, my_runner->render_mesh.point_V_C_N.size() / 3);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glBindVertexArray(0);

            // render for framebuffer map

            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glEnable(GL_DEPTH_TEST);
            //glEnable(GL_BLEND);
            //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glClearColor(-1, -1, -1, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            my_runner->map_shader.use();
            my_runner->map_shader.setMat4("model", model);
            my_runner->map_shader.setMat4("view", view);
            my_runner->map_shader.setMat4("projection", projection);
            glPointSize(10.0);
            glBindVertexArray(my_runner->map_VAO);
            glDrawElements(GL_TRIANGLES, my_runner->render_mesh.V_Idx.size() / 2, GL_UNSIGNED_INT, nullptr);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void delete_collinear(std::vector<IK::Point_2>& poly)
{
    std::vector<IK::Point_2> temp_poly;
    std::set<int> need_to_delete;
    int size = poly.size();
    for (int i = 0; i < size; ++i)
    {
        int p = i, q = (i + 1) % size, r = (i + 2) % size;
        if (CGAL::collinear(poly[p], poly[q], poly[r]))
        {
            need_to_delete.insert(q);
        }
    }
    for (int i = 0; i < size; ++i)
    {
        if (!need_to_delete.count(i))temp_poly.push_back(poly[i]);
    }
    poly.swap(temp_poly);
    temp_poly.clear();
}

void polygon_simplfy(IK_Polygon_2& poly)
{
    auto bbox = poly.bbox();
    auto length = std::max(bbox.xmax() - bbox.xmin(), bbox.ymax() - bbox.ymin());

    std::vector<IK::Point_2> v_poly;
    for (auto p : poly)v_poly.push_back(p);
    
    bool angle_flag = true;
    while (angle_flag)
    {
        angle_flag = false;
        int delete_p = -1;
        delete_collinear(v_poly);

        int size = v_poly.size();
        for (int i = 0; i < v_poly.size(); ++i)
        {
            int p = i, q = (i + 1) % size, r = (i + 2) % size;
            auto v1 = v_poly[r] - v_poly[p];
            auto v2 = v_poly[r] - v_poly[q];
            auto v3 = v_poly[p] - v_poly[q];
            auto cos1 = (CGAL::scalar_product(v1, v2) * CGAL::scalar_product(v1, v2)) / (v1.squared_length() * v2.squared_length());
            auto cos2 = (CGAL::scalar_product(v1, v3) * CGAL::scalar_product(v1, v3)) / (v1.squared_length() * v3.squared_length());
            if (cos1 > 0.98&&cos2>0.98)
            {
                angle_flag = true;
                delete_p = q;
                break;
            }
        }
        if (angle_flag)
        {
            auto it_ = v_poly.begin();
            std::advance(it_, delete_p);
            v_poly.erase(it_);
        }
    }


    bool small_detail_flag = true;

    while (small_detail_flag)
    {
        small_detail_flag = false;
        int delete_p = -1;
        delete_collinear(v_poly);

        int size = v_poly.size();
        for (int i = 0; i < size; ++i)
        {
            int p = i, q = (i + 1) % size;
            auto v1 = v_poly[p];
            auto v2 = v_poly[q];
            auto dis = CGAL::squared_distance(v1, v2);
            dis.set_relative_precision_of_to_double(1e-15);
            auto double_dis = CGAL::to_double(dis);
            auto multiple = length * length / double_dis;
            if (multiple > 2500.0)
            {
                small_detail_flag = true;
                delete_p = q;
                break;
            }
        }
        if (small_detail_flag)
        {
            auto it_ = v_poly.begin();
            std::advance(it_, delete_p);
            v_poly.erase(it_);
        }
    }
    poly.clear();
    for (auto v : v_poly)poly.push_back(v);

}

void pwh_simplify(Polygon_with_holes_2 &pwh)
{
    auto out_ = pwh.outer_boundary();
    int num_of_hole = pwh.number_of_holes();
    for (auto &hole:pwh.holes())
    {
        polygon_simplfy(hole);
    }
}

double compute_angle(inexact_K::Vector_3 n1, inexact_K::Vector_3 n2)
{
    auto cross_v = CGAL::cross_product(n1, n2);
    double radian = atan2(std::sqrt(cross_v.squared_length()), CGAL::scalar_product(n1, n2)); //弧度
    if (cross_v.z() < 0)
    {
        radian = 2 * 3.14 - radian;
    }
    auto angl = radian * 180 / 3.14;
    return angl;
}


std::unique_ptr<IK_Mesh>from_pwh_to_mesh(Polygon_with_holes_2& pwh, IK::Plane_3 unit_plane, IK::Vector_3 length,bool both)
{
    //PS::Squared_distance_cost cost;
    //pwh = PS::simplify(pwh, cost, Stop(0.0025));
    IK_Mesh ans;
    EK_to_IK to_ik;
    std::vector<IK_Polygon_2> polygons;
    IK_Polygon_2 poly;
    for (auto p : pwh.outer_boundary())poly.push_back(to_ik(p));
    if(poly.size()>=3)
        polygons.push_back(poly);
    for (auto hole : pwh.holes())
    {
        IK_Polygon_2 t_p;
        for (auto p : hole)t_p.push_back(to_ik(p));
        if(t_p.size()>=3)
            polygons.push_back(t_p);
    }
    std::vector<std::vector<IK::Point_3>> ik_3d_polygons_1, ik_3d_polygons_2;
    for (auto poly : polygons)
    {
        if (poly.size() < 3)
        {
            std::cout << "<3\n";
            continue;
        }
        std::vector<IK::Point_3> t_poly_1, t_poly_2;
        for (auto p : poly)
        {
            t_poly_1.push_back(unit_plane.to_3d(p) - both*length);
            t_poly_2.push_back(unit_plane.to_3d(p) + length);
        }
        ik_3d_polygons_1.push_back(t_poly_1);
        ik_3d_polygons_2.push_back(t_poly_2);
    }
    std::vector<IK::Point_3> points;
    std::vector<std::vector<size_t>> polygons_soup;
    for (int k = 0; k < ik_3d_polygons_1.size(); ++k)
    {
        auto poly_1 = ik_3d_polygons_1[k];
        auto poly_2 = ik_3d_polygons_2[k];
        for (int j = 0; j < poly_1.size(); ++j)
        {
            std::vector<size_t>soup_poly_1, soup_poly_2;
            if (j != poly_1.size() - 1)
            {
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[j + 1]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[j]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_2[j]);
                polygons_soup.push_back(soup_poly_1);

                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[j]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[j + 1]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_1[j + 1]);
                polygons_soup.push_back(soup_poly_2);

            }
            else
            {
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[0]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[j]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_2[j]);
                polygons_soup.push_back(soup_poly_1);

                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[j]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[0]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_1[0]);
                polygons_soup.push_back(soup_poly_2);
            }
        }
    }//把两边搞好
    // bottom and top
    CDTP cdt;
    
    for (auto poly : polygons)
    {
        if (poly.size() < 3)continue;
        cdt.insert_constraint(poly.vertices_begin(), poly.vertices_end(), true);
    }
    mark_domains(cdt);
    for (Face_handle f : cdt.finite_face_handles())
    {
        if (f->info().in_domain())
        {
            std::vector<std::size_t > _polygon;
            auto v_n = f->ccw(0); auto v_nn = f->ccw(v_n);
            auto p0 = unit_plane.to_3d(cdt.point(f->vertex(0)));
            auto p1 = unit_plane.to_3d(cdt.point(f->vertex(v_n)));
            auto p2 = unit_plane.to_3d(cdt.point(f->vertex(v_nn)));
            _polygon.push_back(points.size());
            points.push_back(p0);
            _polygon.push_back(points.size());
            points.push_back(p1);
            _polygon.push_back(points.size());
            points.push_back(p2);
            polygons_soup.push_back(_polygon);
            _polygon.clear();
            _polygon.push_back(points.size());
            points.push_back(p0 + length);
            _polygon.push_back(points.size());
            points.push_back(p1 + length);
            _polygon.push_back(points.size());
            points.push_back(p2 + length);
            polygons_soup.push_back(_polygon);
        }
    }
    if (polygons_soup.size())
    {
        PMP::repair_polygon_soup(points, polygons_soup);
        PMP::orient_polygon_soup(points, polygons_soup);
        PMP::polygon_soup_to_polygon_mesh(points, polygons_soup, ans);
        if (!CGAL::is_closed(ans))
        {
            if (!PMP::is_polygon_soup_a_polygon_mesh(polygons_soup))
                std::cout<<"not a polygon mesh\n";
            PMP::stitch_borders(ans);
            if (!CGAL::is_closed(ans))
            {
                std::cout << "pwh still no closed\n"; 
                const std::string path = SAVE_PATH;
                const std::string fullpath = path + "no_close_mesh" + std::to_string(points.size()) + ".off";
                std::ofstream out(fullpath);
                out << (ans);
            }
        }
        PMP::orient_to_bound_a_volume(ans);
    }
    cdt.clear();
    return std::make_unique<IK_Mesh>(ans);
}

std::unique_ptr<Nef_polyhedron> poly_to_polyhedron(Polygon_2 init_poly, IK::Plane_3 unit_plane, IK::Vector_3 length)
{
    Nef_polyhedron ans;
    Polyhedron Ph;
}

std::unique_ptr<IK_Mesh>from_poly_to_mesh(Polygon_2 init_poly, IK::Plane_3 unit_plane, IK::Vector_3 length ,bool both_side)
{
    IK_Mesh ans;
    EK_to_IK to_ik;
    std::vector<IK_Polygon_2> polygons;
    IK_Polygon_2 poly;
    for (auto p : init_poly)poly.push_back(p);
    polygons.push_back(poly);
    double ratio = 1.0;
    if (!both_side)ratio = 0.0;
    std::vector<std::vector<IK::Point_3>> ik_3d_polygons_1, ik_3d_polygons_2;
    for (auto poly : polygons)
    {
        std::vector<IK::Point_3> t_poly_1, t_poly_2;
        for (auto p : poly)
        {
            t_poly_1.push_back(unit_plane.to_3d(p) - length*ratio);
            t_poly_2.push_back(unit_plane.to_3d(p) + length);
        }
        ik_3d_polygons_1.push_back(t_poly_1);
        ik_3d_polygons_2.push_back(t_poly_2);
    }
    std::vector<IK::Point_3> points;
    std::vector<std::vector<size_t>> polygons_soup;
    for (int k = 0; k < ik_3d_polygons_1.size(); ++k)
    {
        auto poly_1 = ik_3d_polygons_1[k];
        auto poly_2 = ik_3d_polygons_2[k];

        for (int j = 0; j < poly_1.size(); ++j)
        {
            std::vector<size_t>soup_poly_1, soup_poly_2;
            if (j != poly_1.size() - 1)
            {
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[j + 1]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[j]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_2[j]);
                polygons_soup.push_back(soup_poly_1);

                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[j]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[j + 1]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_1[j + 1]);
                polygons_soup.push_back(soup_poly_2);

            }
            else
            {
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[0]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_1[j]);
                soup_poly_1.push_back(points.size());
                points.push_back(poly_2[j]);
                polygons_soup.push_back(soup_poly_1);

                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[j]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_2[0]);
                soup_poly_2.push_back(points.size());
                points.push_back(poly_1[0]);
                polygons_soup.push_back(soup_poly_2);
            }
        }
    }//把两边搞好
    // bottom and top
    CDTP cdt;
    for (auto poly : polygons)
    {
        if(poly.size()>=3)
            cdt.insert_constraint(poly.vertices_begin(), poly.vertices_end(), true);
    }
    mark_domains(cdt);
    for (Face_handle f : cdt.finite_face_handles())
    {
        if (f->info().in_domain())
        {
            std::vector<std::size_t > _polygon;
            auto v_n = f->ccw(0); auto v_nn = f->ccw(v_n);
            auto p0 = unit_plane.to_3d(cdt.point(f->vertex(0)));
            auto p1 = unit_plane.to_3d(cdt.point(f->vertex(v_n)));
            auto p2 = unit_plane.to_3d(cdt.point(f->vertex(v_nn)));
            _polygon.push_back(points.size());
            points.push_back(p0 - length * ratio);
            _polygon.push_back(points.size());
            points.push_back(p1 - length * ratio);
            _polygon.push_back(points.size());
            points.push_back(p2 - length * ratio);
            polygons_soup.push_back(_polygon);
            _polygon.clear();
            _polygon.push_back(points.size());
            points.push_back(p0 + length);
            _polygon.push_back(points.size());
            points.push_back(p1 + length);
            _polygon.push_back(points.size());
            points.push_back(p2 + length);
            polygons_soup.push_back(_polygon);
        }
    }

    std::cout << "Before reparation, the soup has " << points.size() << " vertices and " << polygons_soup.size() << " faces" << std::endl;
    PMP::repair_polygon_soup(points, polygons_soup);
    std::cout << "After reparation, the soup has " << points.size() << " vertices and " << polygons_soup.size() << " faces" << std::endl;
    PMP::orient_polygon_soup(points, polygons_soup);
    PMP::polygon_soup_to_polygon_mesh(points, polygons_soup, ans);
    if (!CGAL::is_closed(ans))
    {
        if (!PMP::is_polygon_soup_a_polygon_mesh(polygons_soup))
            std::cout << "not a polygon mesh\n";
        PMP::stitch_borders(ans);
        if (!CGAL::is_closed(ans))
            std::cout << "poly still no closed\n";
    }
    PMP::orient_to_bound_a_volume(ans);
    cdt.clear();
    return std::make_unique<IK_Mesh>(ans);
}

bool is_very_crease_edge(IK_Mesh& mesh, it_E e) {
    if (mesh.is_border(e))return true;
    auto h1 = e.halfedge();
    auto p1 = mesh.point(mesh.source(h1));
    auto p2 = mesh.point(mesh.source(mesh.next(h1)));
    auto p3 = mesh.point(mesh.source(mesh.next(mesh.next(h1))));
    auto n1 = CGAL::cross_product(p2 - p1, p3 - p2);

    auto h2 = mesh.opposite(h1);
    p1 = mesh.point(mesh.source(h2));
    p2 = mesh.point(mesh.source(mesh.next(h2)));
    p3 = mesh.point(mesh.source(mesh.next(mesh.next(h2))));
    auto n2 = CGAL::cross_product(p2 - p1, p3 - p2);

    auto angle_2 = (n1 * n2) * (n1 * n2) / (n1.squared_length() * n2.squared_length());
    angle_2.set_relative_precision_of_to_double(1e-10);
    auto angle = CGAL::to_double(angle_2);
    bool is_crease = (angle < 0.25);
    return is_crease;
}

bool is_crease_edge(IK_Mesh& mesh, CGAL::SM_Edge_index e)
{
    auto h1 = e.halfedge();
    auto n1 = PMP::compute_face_normal(mesh.face(h1), mesh);
    auto h2 = mesh.opposite(h1);
    auto n2 = PMP::compute_face_normal(mesh.face(h2), mesh);
    auto n1_sq = n1.squared_length();
    auto n2_sq = n2.squared_length();
    if (CGAL::to_double(n1_sq) < 1e-10 || CGAL::to_double(n2_sq) < 1e-10)return true;
    auto angle_2 = (n1 * n2) * (n1 * n2) / (n1.squared_length() * n2.squared_length());
    angle_2.set_relative_precision_of_to_double(1e-10);
    auto angle = CGAL::to_double(angle_2);
    bool is_crease = (angle < 0.999);
    return is_crease;
}

void my_new_simplify(IK_Mesh& surface)
{
    PMP::keep_largest_connected_components(surface, 1);
    surface.collect_garbage();
    if (!CGAL::is_closed(surface) || !PMP::does_bound_a_volume(surface))
    {
        std::cout << "don't satisfy simplify requirement\n";
        return;
    }
    IK_Mesh new_mesh;

    surface.add_property_map<CGAL::SM_Vertex_index, bool>("is_deleted", false);
    surface.add_property_map<CGAL::SM_Edge_index, bool>("is_deleted", false);
    //surface.add_property_map<CGAL::SM_Face_index, bool>("is_deleted", false);

    //-------------------------------------delete crease_edge---------------
    for (auto e : surface.edges())
    {
        if (!is_crease_edge(surface, e))
            surface.property_map<CGAL::SM_Edge_index, bool>
            ("is_deleted").first[e] = true;
    }

    //------------------------------------delete vertices without edge
    //for (auto p : surface.vertices())
    //{
    //    bool is_edge_on_vertex = false;
    //    for (auto e : surface.halfedges_around_target(surface.halfedge(p)))
    //    {
    //        if (!surface.property_map<CGAL::SM_Edge_index, bool>
    //            ("is_deleted").first[surface.edge(e)])is_edge_on_vertex = true;
    //    }
    //    if (!is_edge_on_vertex) surface.property_map<CGAL::SM_Vertex_index, bool>
    //        ("is_deleted").first[p] = true;
    //}

    ////------------------------------------delete faces without three edges
    //for (auto f : surface.faces())
    //{
    //    bool need_to_delete = false;
    //    for (auto e : surface.halfedges_around_face(surface.halfedge(f)))
    //    {
    //        if (surface.property_map<CGAL::SM_Edge_index, bool>
    //            ("is_deleted").first[surface.edge(e)])
    //        {
    //            need_to_delete = true;
    //            break;
    //        }

    //    }
    //    if (need_to_delete)
    //        surface.property_map<CGAL::SM_Face_index, bool>
    //        ("is_deleted").first[f] = true;

    //}
    ////merge two link faces without a egde between them
    //

    ////    delete vertices with less than two crease_edges
    //for (auto p : surface.vertices())
    //{
    //    if (surface.property_map<CGAL::SM_Vertex_index, bool>
    //        ("is_deleted").first[p])continue;
    //    int count = 0;
    //    for (auto e : surface.halfedges_around_target(surface.halfedge(p)))
    //    {
    //        if (!surface.property_map<CGAL::SM_Edge_index, bool>
    //            ("is_deleted").first[surface.edge(e)])++count;
    //    }
    //    if (count <= 2)
    //        surface.property_map<vertex_descriptor, bool>("is_deleted").first[p] = true;
    //}
    //delete edge without point
    //for (auto edge : surface.edges())
    //{
    //    auto v0 = surface.vertex(edge, 0);
    //    auto v1 = surface.vertex(edge, 1);
    //    if(surface.property_map<CGAL::SM_Vertex_index, bool>
    //        ("is_deleted").first[v0] && surface.property_map<CGAL::SM_Vertex_index, bool>
    //        ("is_deleted").first[v1])
    //        surface.property_map<CGAL::SM_Edge_index, bool>
    //        ("is_deleted").first[edge] = true;
    //}

    //IK_Mesh::Property_map<halfedge_descriptor, std::pair<IK::Point_3, IK::Point_3> > constrained_halfedges;
    //constrained_halfedges = surface.add_property_map<halfedge_descriptor, std::pair<IK::Point_3, IK::Point_3> >("h:vertices").first;
    //std::size_t constrain_edges = 0;
    //for (auto h_edge: surface.halfedges())
    //{
    //    auto edge = surface.edge(h_edge);
    //    if (surface.property_map<CGAL::SM_Edge_index, bool>
    //        ("is_deleted").first[edge])
    //    {
    //        constrained_halfedges[h_edge] = std::make_pair(surface.point(surface.source(h_edge)),
    //            surface.point(surface.target(h_edge)));
    //        ++constrain_edges;
    //    }
    //}
    //std::cout << "constrain_edges'num" << constrain_edges << '\n';

    PMP::isotropic_remeshing(
        CGAL::faces(surface),
        1e100, //a value larger than bbox
        surface,
        CGAL::Polygon_mesh_processing::parameters::edge_is_constrained_map(surface.property_map<CGAL::SM_Edge_index, bool>
            ("is_deleted").first)
    );
    surface.remove_all_property_maps();
    //SMS::Count_stop_predicate<IK_Mesh> stop(0);
    //Border_is_constrained_edge_map bem(surface);
    //int r = SMS::edge_collapse(surface, stop,
    //    CGAL::parameters::edge_is_constrained_map(bem)
    //    .get_placement(Placement(bem)));


}



bool simplify(IK_Mesh& surface)
{
    
    if (!CGAL::is_closed(surface))return false;
    PMP::keep_largest_connected_components(surface, 1);
    IK_Mesh new_mesh;
    surface.add_property_map<CGAL::SM_Vertex_index, bool>("is_deleted", false);
    surface.add_property_map<CGAL::SM_Edge_index, bool>("is_deleted", false);
    if (surface.number_of_faces() == 0)return false;
    //-------------------------------------delete crease_edge---------------
    for (auto e : surface.edges())
    {
        if (!is_crease_edge(surface, e))
            surface.property_map<CGAL::SM_Edge_index, bool>
            ("is_deleted").first[e] = true;
    } 
    //------------------------------------delete vertices without edge
    for (auto p : surface.vertices())
    {
        bool is_edge_on_vertex = false;
        for (auto e : surface.halfedges_around_target(surface.halfedge(p)))
        {
            if (!surface.property_map<CGAL::SM_Edge_index, bool>
                ("is_deleted").first[surface.edge(e)])is_edge_on_vertex = true;
        }
        if (!is_edge_on_vertex) surface.property_map<CGAL::SM_Vertex_index, bool>
            ("is_deleted").first[p] = true;
    }

    //merge two link faces without a egde between them
    m.init();
    for (auto face : surface.faces())
    {
        if (m.p[face.idx()] != -1)continue;
        m.p[face.idx()] = face.idx();
        std::vector<CGAL::SM_Face_index> q;
        q.push_back(face);
        while (!q.empty())
        {
            auto f = q.back();
            q.pop_back();
            for (auto h_e : surface.halfedges_around_face(surface.halfedge(f)))
            {
                auto op_e = surface.opposite(h_e);
                auto op_f = surface.face(op_e);
                if (surface.property_map<CGAL::SM_Edge_index, bool>
                    ("is_deleted").first[surface.edge(h_e)])
                {
                    if (m.p[(int)op_f.idx()] != -1 && m.p[(int)op_f.idx()] != face.idx())std::cout << "error\n";
                    if (m.p[(int)op_f.idx()] == -1)
                    {
                        q.push_back(op_f);
                        m.merge((int)f.idx(), (int)op_f.idx());
                    }
                }
            }
        }
    }

    //find out regions(linked faces)'s edges whice are not deleted

    std::vector<std::vector<CGAL::SM_Halfedge_index>> Regions;
    std::set<int> hash_set;
    std::map<int, int> id_to_Regionsid;
    int count = 0;
    for (auto f : surface.faces())
    {
        int id = m.find((int)f.idx());
        if (hash_set.find(id) == hash_set.end())
        {
            std::vector<CGAL::SM_Halfedge_index> region;
            Regions.push_back(region);
            hash_set.insert(id);
            id_to_Regionsid[id] = count;
            ++count;
        }
        int R_id = id_to_Regionsid[id];
        for (auto e : surface.halfedges_around_face(surface.halfedge(f)))
        {
            if (!surface.property_map<CGAL::SM_Edge_index, bool>
                ("is_deleted").first[surface.edge(e)])
                Regions[R_id].push_back(e);
        }
    }

    //    delete vertices with less than two crease_edges
    for (auto p : surface.vertices())
    {
        if (surface.property_map<CGAL::SM_Vertex_index, bool>
            ("is_deleted").first[p])continue;
        int count = 0;
        for (auto e : surface.halfedges_around_target(surface.halfedge(p)))
        {
            if (!surface.property_map<CGAL::SM_Edge_index, bool>
                ("is_deleted").first[surface.edge(e)])++count;
        }
        if (count == 2)
            surface.property_map<CGAL::SM_Vertex_index, bool>("is_deleted").first[p] = true;
    }

    //get new_mesh

    std::vector<std::vector<std::size_t>>polygon_soup;
    std::vector<IK::Point_3>points;
    for (int i = 0; i < Regions.size(); ++i)
    {
        std::vector<std::vector<CGAL::SM_Vertex_index>> _3d_polygons;
        auto region = Regions[i];
        int edges_size = region.size();
        if (edges_size < 3)
        {
            std::cout << "has a region less than three edges\n";
        }
        bool* had_in_one_loop = (bool*)malloc(sizeof(bool) * edges_size);
        memset(had_in_one_loop, 0, edges_size * sizeof(bool));
        while (true)
        {
            bool* had_in_this_loop = (bool*)malloc(sizeof(bool) * edges_size);
            memset(had_in_this_loop, 0, edges_size * sizeof(bool));
            std::vector<CGAL::SM_Vertex_index> loop;
            int begin_edge = -1;
            for (int j = 0; j < edges_size; ++j)
                if (had_in_one_loop[j] == 0)
                {
                    begin_edge = j;
                    break;
                }
            if (begin_edge == -1)break;
            auto v_n = surface.target(region[begin_edge]);
            had_in_one_loop[begin_edge] = true;
            had_in_this_loop[begin_edge] = true;
            loop.push_back((surface.source(region[begin_edge])));
            while (true)
            {
                bool is_still_edge_in_loop = false;
                for (int j = 0; j < edges_size; ++j)
                {
                    if (had_in_this_loop[j])continue;
                    if (surface.source(region[j]) == v_n)
                    {
                        loop.push_back(v_n);
                        had_in_one_loop[j] = true;
                        had_in_this_loop[j] = true;
                        v_n = surface.target(region[j]);
                        is_still_edge_in_loop = true;
                        break;
                    }
                }
                if (!is_still_edge_in_loop)
                {
                    break;
                }
            }
            free(had_in_this_loop);
            _3d_polygons.push_back(loop);
            if (loop.size() < 3)
            {
                std::cout << edges_size << "edges size\n";
                std::cout << "loop size <3: " << loop.size() << '\n';
            }
        }

        std::vector<IK::Point_3>points_plane;
        for (auto poly : _3d_polygons)
        {
            for (auto p : poly)
                points_plane.push_back(surface.point(p));
        }
        if (points_plane.size() < 3)
        {
            std::cout << points_plane.size() << "\n all_points in polygon3";
        }
        CGAL::Plane_3<IK> plane;
        linear_least_squares_fitting_3(
            points_plane.begin(),
            points_plane.end(),
            plane,
            CGAL::Dimension_tag<0>());//...


        for (int j = 0; j < _3d_polygons.size(); ++j)
        {
            for (auto it = _3d_polygons[j].begin(); it != _3d_polygons[j].end(); )
            {
                if (surface.property_map<CGAL::SM_Vertex_index, bool>("is_deleted").first[*it])
                    it = _3d_polygons[j].erase(it);
                else ++it;
            }
        }

        std::vector<IK_Polygon_2>_2d_polygons;
        int num_of_v = 0;
        for (auto _3d_poly : _3d_polygons)
        {
            IK_Polygon_2 poly;
            for (auto p : _3d_poly)
            {
                ++num_of_v;
                poly.push_back(CDT_Point(plane.to_2d(surface.point(p))));
            }
            _2d_polygons.push_back(poly);

        }

        CDTP cdt;
        for (auto poly : _2d_polygons)
        {
            if (poly.size() >= 1)
                cdt.insert_constraint(poly.vertices_begin(), poly.vertices_end(), true);
        }
        mark_domains(cdt);
        for (Face_handle f : cdt.finite_face_handles())
        {
            if (f->info().in_domain())
            {
                std::vector<std::size_t > _polygon;

                auto v_n = cdt.ccw(0);
                auto v_nn = cdt.ccw(v_n);
                auto p0 = cdt.point(f->vertex(0));
                auto p1 = cdt.point(f->vertex(v_n));
                auto p2 = cdt.point(f->vertex(v_nn));
                size_t idx_poly_0 = 0, idx_p0 = 0, idx_poly_1 = 0, idx_p1 = 0, idx_poly_2 = 0, idx_p2 = 0;
                for (size_t idx_poly = 0; idx_poly < _2d_polygons.size(); ++idx_poly)
                {
                    size_t idx_p = 0;
                    for (auto p : _2d_polygons[idx_poly])
                    {
                        if (p == p0)idx_p0 = idx_p, idx_poly_0 = idx_poly + 1;
                        if (p == p1)idx_p1 = idx_p, idx_poly_1 = idx_poly + 1;
                        if (p == p2)idx_p2 = idx_p, idx_poly_2 = idx_poly + 1;
                        ++idx_p;

                        if (idx_poly_0 != 0 && idx_poly_1 != 0 && idx_poly_2 != 0)break;
                    }
                    if (idx_poly_0 != 0 && idx_poly_1 != 0 && idx_poly_2 != 0)break;
                }
                if (idx_poly_0 == 0 || idx_poly_1 == 0 || idx_poly_2 == 0)
                {
                    puts("new point?");
                    surface.remove_all_property_maps();
                    return false;
                }
                _polygon.push_back(points.size());
                points.push_back(surface.point(_3d_polygons[idx_poly_0 - 1][idx_p0]));
                _polygon.push_back(points.size());
                points.push_back(surface.point(_3d_polygons[idx_poly_1 - 1][idx_p1]));
                _polygon.push_back(points.size());
                points.push_back(surface.point(_3d_polygons[idx_poly_2 - 1][idx_p2]));
                polygon_soup.push_back(_polygon);
            }
        }
        free(had_in_one_loop);
    }
    
    surface.clear();
    PMP::repair_polygon_soup(points, polygon_soup);
    PMP::orient_polygon_soup(points, polygon_soup);
    PMP::polygon_soup_to_polygon_mesh(points, polygon_soup, surface);
    if (!CGAL::is_closed(surface))
    {
        if (!PMP::is_polygon_soup_a_polygon_mesh(polygon_soup))
            return false;
        PMP::stitch_borders(surface);
        if (!CGAL::is_closed(surface))
        {
            return false;
        }
    }
    PMP::orient_to_bound_a_volume(surface);
    return true;
}
//---------------------------------------------------------------

std::string Runner::window_select_file()
{
    OPENFILENAME ofn;
    char szFile[300];
    std::string temp = "All\0*.*\0Text\0*.TXT\0";
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = (LPSTR)(LPCSTR)szFile;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = (LPSTR)(LPCSTR)temp.c_str();
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    if (GetOpenFileName(&ofn))
        return ofn.lpstrFile;
    else
        return "";
}
glm::vec2 transform_mouse(glm::vec2 in, int win_width, int win_height)
{
    return glm::vec2(in.x * 2.f / win_width - 1.f, 1.f - 2.f * in.y / win_height);
}
inline Vec3 compute_normal(Vec3 p1,Vec3 p2,Vec3 p3)
{
    double na = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
    double nb = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
    double nc = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    return Vec3(na, nb, nc);
}
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}
void Runner::my_ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{//o! matrix should set before each render
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    camera.zoom(yoffset);
}
void Runner::frame_buffer_callback(GLFWwindow* window, int w, int h)
{
    glViewport(0, 0, w, h);
    SCR_WIDTH = w;
    SCR_HEIGHT = h;
}
std::unique_ptr<IK_Mesh> Runner::to_surface_mesh(libgl_mesh& mesh)
{
    IK_Mesh ans;
    int v_size = mesh.V.rows();
    std::vector<CGAL::SM_Vertex_index> SVis;
    for (int i = 0; i < v_size; ++i)
    {
        auto t =ans.add_vertex(IK::Point_3(mesh.V(i, 0), mesh.V(i, 1), mesh.V(i, 2)));
        SVis.push_back(t);
    }
    int f_size = mesh.F.rows();
    for (int i = 0; i < f_size; ++i)
    {
        ans.add_face(SVis[mesh.F(i, 0)], 
            SVis[mesh.F(i, 1)],
            SVis[mesh.F(i, 2)]);
    }

    return std::make_unique<IK_Mesh>(ans);
}
std::unique_ptr<libgl_mesh> Runner::to_libigl_mesh(IK_Mesh& mesh)
{
    libgl_mesh ans;
    int cnt = 0;
    ans.V.resize(mesh.number_of_vertices(), 3);
    std::unordered_map<int, int> vertex_idx_to_matrix_idx;
    for (auto& vertex : mesh.vertices())
    {
        vertex_idx_to_matrix_idx[(int)vertex.idx()] = cnt;
        auto p = mesh.point(vertex);
        auto x = p.x();
        auto y = (p.y());
        auto z = (p.z());
        x.set_relative_precision_of_to_double(1e-15);
        y.set_relative_precision_of_to_double(1e-15);
        z.set_relative_precision_of_to_double(1e-15);
        ans.V(cnt, 0) = CGAL::to_double(x);
        ans.V(cnt, 1) = CGAL::to_double(y);
        ans.V(cnt, 2) = CGAL::to_double(z);
        ++cnt;
    }
    cnt = 0;
    ans.F.resize(mesh.number_of_faces(), 3);
    for (auto& face : mesh.faces())
    {
        auto face_half_edge = mesh.halfedge(face);
        if (mesh.vertices_around_face(face_half_edge).size() != 3) {
            puts("not a mesh");
        }
        int v_x = 0;
        for (auto v_id : mesh.vertices_around_face(face_half_edge))
        {
            ans.F(cnt, v_x) = vertex_idx_to_matrix_idx[(int)v_id.idx()];
            ++v_x;
        }
        ++cnt;
    }

    return std::make_unique<libgl_mesh>(ans);
}
std::unique_ptr<Polyhedron> Runner::to_Polyhedron(IK_Mesh& mesh)
{
    const auto libigl_mesh = *to_libigl_mesh(mesh);
    Polyhedron ans;
    return std::make_unique<Polyhedron>(ans);
}
void Runner::get_matrices()
{
    E_V_matrix.clear();
    F_E_matrix.clear();
    for (auto edge : mesh.edges())
    {
        auto v1 = mesh.vertex(edge, 0);
        auto v2 = mesh.vertex(edge, 1);
        E_V_matrix.push_back({ edge.idx(),v1.idx() });
        E_V_matrix.push_back({ edge.idx(),v2.idx() });
    }

    for (auto face : mesh.faces())
        for (auto h_edge : mesh.halfedges_around_face(mesh.halfedge(face)))
        {
            auto edge = mesh.edge(h_edge);
            F_E_matrix.push_back({ face.idx(),edge.idx()});
        }
}
void Runner::build_bool_vec()
{
    vertices_vec.clear();
    edges_vec.clear();
    faces_vec.clear();
    vertices_vec.resize(mesh.number_of_vertices());
    edges_vec.resize(mesh.number_of_edges());
    faces_vec.resize(mesh.number_of_faces());

    for (auto id : selected_vertex)
        vertices_vec[id.idx()] = 1;
    for (auto id : selected_edge)
        edges_vec[id.idx()] = 1;
    for (auto id : selected_face)
        faces_vec[id.idx()] = 1;
}
void Runner::get_input()
{
    input_.clear();
    for (auto id : selected_vertex)
    {
        simplex temp;
        temp.id = id.idx();
        temp.level = 1;
        input_.push_back(temp);
    }
    for (auto id : selected_edge)
    {
        simplex temp;
        temp.id = id.idx();
        temp.level = 2;
        input_.push_back(temp);
    }
    for (auto id : selected_face)
    {
        simplex temp;
        temp.id = id.idx();
        temp.level = 3;
        input_.push_back(temp);
    }
}
void Runner::clear_result()
{
    result_.clear();
}
void Runner::generate_Visual_hull_views()
{
    int num_of_views = 15;
    double const PI = 3.14159265;
    std::vector<K::Vector_3> dirs;
    for (int i = 1; i < num_of_views; ++i)
    {
        auto phi = std::acos(-1.0 + (2.0 * i - 1.0) / num_of_views);
        auto theta = std::sqrt(num_of_views * PI) * phi;
        auto dir = -K::Vector_3(std::cos(theta) * std::sin(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(phi));
        dirs.push_back(dir);
    }
    std::vector<plane_3> planes(dirs.size());
    std::vector<Polygon_with_holes_2> pwhs(dirs.size());
    std::vector<std::pair<double, int>> area_difference_index(dirs.size());
#pragma omp parallel for
    for (int i = 0; i < dirs.size(); ++i)
    {
        EK_to_IK to_ik;
        IK_to_EK to_ek;
        std::vector<Polygon_2> polygons;
        auto projection_plane = to_ik(plane_3(Point_3(0, 0, 3), dirs[i]));
        auto unit_plane = get_unit_plane(projection_plane);
        // try keep just one polygon_with_hole
        for (auto face : ik_mesh.faces())
        {
            Polygon_2 poly;
            std::vector<IK::Point_2> after_project_points;
            for (auto vertex : ik_mesh.vertices_around_face(ik_mesh.halfedge(face)))
            {
                auto point = ik_mesh.point(vertex);
                IK::Point_3 pro_point = unit_plane.projection(point);
                after_project_points.push_back(unit_plane.to_2d(pro_point));//to_2d !
            }
            IK::Triangle_2 tri(after_project_points[0], after_project_points[1], after_project_points[2]);
            if (tri.is_degenerate())
                continue;
            for (auto p : after_project_points)
                poly.push_back(to_ek(p));
            if (!poly.is_counterclockwise_oriented())
                poly.reverse_orientation();
            polygons.push_back(poly);
        }

        std::vector<Polygon_with_holes_2> p_w_h;
        CGAL::join(polygons.begin(), polygons.end(), std::back_inserter(p_w_h));
        auto poly_t = p_w_h[0];
        for (int i = 1; i < p_w_h.size(); ++i)
            if (p_w_h[i].outer_boundary().size() > poly_t.outer_boundary().size())
                poly_t = p_w_h[i];
        double area_diff =
            (poly_t.bbox().xmax() - poly_t.bbox().xmin()) *
            (poly_t.bbox().ymax() - poly_t.bbox().ymin());
        area_diff -= CGAL::to_double(poly_t.outer_boundary().area());
        for (auto& hole : poly_t.holes())area_diff += CGAL::to_double(hole.area());
        planes[i] = unit_plane;
        pwhs[i] = poly_t;
        area_difference_index[i] = std::make_pair(area_diff, i);
    }

    std::sort(area_difference_index.begin(), area_difference_index.end(), std::greater<std::pair<double, int>>());

    for (int i = 0; i < 10; ++i)
    {
        const int& index = area_difference_index[i].second;
        Visual_hull_view V(planes[index], pwhs[index]);
        V_hull_views.push_back(V);
    }
}
void Runner::detect_planes()
{
    const Face_range face_range = faces(mesh);
    std::cout <<
        "* polygon mesh with "
        << face_range.size() <<
        " faces is loaded"
        << std::endl;
    // Default parameter values for the data file polygon_mesh.off.
    const FT          max_distance_to_plane = FT(0.05);
    const FT          max_accepted_angle = FT(25);
    const std::size_t min_region_size = 3;
    // Create instances of the classes Neighbor_query and Region_type.
    Neighbor_query neighbor_query(mesh);
    const Vertex_to_point_map vertex_to_point_map(
        get(CGAL::vertex_point, mesh));
    Region_type region_type(
        mesh,
        max_distance_to_plane, max_accepted_angle, min_region_size,
        vertex_to_point_map);
    // Sort face indices.
    Sorting sorting(
        mesh, neighbor_query,
        vertex_to_point_map);
    sorting.sort();
    // Create an instance of the region growing class.
    Region_growing region_growing(
        face_range, neighbor_query, region_type,
        sorting.seed_map());
    // Run the algorithm.
    Regions regions;
    region_growing.detect(std::back_inserter(regions));
    // Print the number of found regions.
    std::cout << "* " << regions.size() <<
        " regions have been found"
        << std::endl;
    // Save the result in a file only if it is stored in CGAL::Surface_mesh.

    using Face_index = typename Polygon_mesh::Face_index;
    // Save the result to a file in the user-provided path if any.
    srand(static_cast<unsigned int>(time(nullptr)));

    bool created;
    typename Polygon_mesh::template Property_map<Face_index, Color> face_color;
    boost::tie(face_color, created) =
        mesh.template add_property_map<Face_index, Color>(
            "f:color", Color(0, 0, 0));
    if (!created) {
        std::cout << std::endl <<
            "region_growing_on_polygon_mesh example finished"
            << std::endl << std::endl;
    }
    const std::string path = SAVE_PATH;
    const std::string fullpath = path + "regions_polygon_mesh.off";
    std::ofstream out(fullpath);
    // Iterate through all regions.
    for (const auto& region : regions) {
        // Generate a random color.
        const Color color(
            static_cast<unsigned char>(rand() % 256),
            static_cast<unsigned char>(rand() % 256),
            static_cast<unsigned char>(rand() % 256));
        // Iterate through all region items.
        using size_type = typename Polygon_mesh::size_type;
        for (const auto index : region)
            face_color[Face_index(static_cast<size_type>(index))] = color;
    }
    out << mesh;
    out.close();
    std::cout <<
        "* polygon mesh is saved in "
        << fullpath << std::endl;

    std::cout << std::endl <<
        "region_growing_on_polygon_mesh example finished"
        << std::endl << std::endl;

    //--regions_plane

    for (auto& region : regions)
    {
        EK_to_IEK to_iek;
        IEK_to_IK to_ek;
        std::vector<inexact_K::Triangle_3> mesh_tris;
        inexact_K::Vector_3 plane_normal(0, 0, 0);
        Regions_plane temp;
        temp.area = 0;
        using size_type = typename Polygon_mesh::size_type;
        for (auto id : region)temp.face_its.push_back(CGAL::SM_Face_index(static_cast<size_type>(id)));
        for (auto& fd : temp.face_its)
        {
            temp.fds_set.insert(fd);
            detected_fds.insert(fd);
        }
        for (const auto& fd : temp.face_its)
        {
            if (PMP::is_degenerate_triangle_face(fd, mesh))
                continue;
            std::vector<inexact_K::Point_3> points;
            for (const auto& vd : mesh.vertices_around_face(mesh.halfedge(fd)))
            {
                points.push_back(to_iek(mesh.point(vd)));
            }
           
            mesh_tris.emplace_back(points[0],points[1],points[2]);
            const auto& tri=mesh_tris.back();
            plane_normal += CGAL::unit_normal(tri[0], tri[1], tri[2]);
            temp.area += std::sqrt(tri.squared_area());
        }
        inexact_K::Plane_3 inexact_plane;
        linear_least_squares_fitting_3(
            mesh_tris.begin(),
            mesh_tris.end(),
            inexact_plane,
            CGAL::Dimension_tag<2>());
        temp.plane = to_ek(inexact_plane);
        auto s_p = CGAL::scalar_product(to_iek(temp.plane.orthogonal_vector()), plane_normal);
        if (s_p < 0)temp.plane = temp.plane.opposite();
        regions_plane.push_back(temp);
        carve_planes.push_back(temp);
    }
    {
    std::cout << "regions_plane_size" << regions_plane.size() << '\n';

    EK_to_IEK to_iek;
    auto size = regions_plane.size();
    for (int i = 0; i < size; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            ms.normals[i] = regions_plane[i].plane.orthogonal_vector();
            auto plane1 = regions_plane[i].plane;
            auto plane2 = regions_plane[j].plane;
            
            auto n1 = to_iek(plane1.orthogonal_vector());
            auto n2 = to_iek(plane2.orthogonal_vector());

            auto angle = compute_angle(n1, n2);
            if (angle < 30)ms.merge(i, j);
        }
        //std::cout << i << '\n';
    }

    for (int i = 0; i < regions_plane.size(); ++i)
    {
        areas[ms.find(i)] += regions_plane[i].area;
    }
    for (int i = 0; i < 10000; ++i)
    {
        if (areas[i] > 0.005)
        {
            normal_area temp;
            temp.normal = ms.normals[i];
            temp.area = areas[i];
            normal_areas.push_back(temp);
        }
    }
    //produce=====view=====direction
    std::cout << "normal_areas_size" << normal_areas.size() << '\n';
    for(int i=0;i<normal_areas.size();++i)
        for (int j = 0; j < normal_areas.size(); ++j)
        {
            auto dir = CGAL::cross_product(normal_areas[i].normal, normal_areas[j].normal);
            view_direction v_d;
            v_d.Dir = dir;
            v_d.Dir = K::Vector_3(v_d.Dir.x(), 0.0f, v_d.Dir.z());
            v_d.area = normal_areas[i].area + normal_areas[j].area;
            if (CGAL::to_double(v_d.Dir.x()) < 1e-5 && CGAL::to_double(v_d.Dir.z())<1e-5)continue;
            All_views.push_back(v_d);
        }


    for (int i = 0; i < All_views.size(); ++i)
        for (int j = i + 1; j < All_views.size(); ++j)
        {
            view_set.normals[i] = All_views[i].Dir;
            auto dir1 = All_views[i].Dir;
            auto dir2 = All_views[j].Dir;
            auto n1 = to_iek(dir1);
            auto n2 = to_iek(dir2);
            if (std::fabs(compute_angle(n1, n2)) < 5)
            {
                view_set.merge(i, j);
            }
        }
    for (int i = 0; i < All_views.size(); ++i)
    {
        if (view_set.p[i] != i)
            continue;
         after_merged_All_views.push_back(All_views[i]);
    }
    All_views.clear();
    All_views.assign(after_merged_All_views.begin(), after_merged_All_views.end());
    std::sort(All_views.begin(), All_views.end());
    std::cout << "all_size:" << All_views.size() << std::endl;
    int upper_bound = std::min(150, (int)All_views.size());

    if (upper_bound % 2)--upper_bound;
    for (int i = 0; i < upper_bound; ++i) {
        Visual_hull_views.push_back(All_views[i]);
    }
}
    // old method generate v_hull
    
    // carve_planes

    std::sort(carve_planes.begin(), carve_planes.end());
    std::cout << "carve_planes_size:" << carve_planes.size() << '\n';
    int max_carve_planes = std::min(Config::Path::get().planes, (int)carve_planes.size());
    carve_planes.assign(carve_planes.begin(), carve_planes.begin() + max_carve_planes);
    for (auto plane : carve_planes)std::cout << plane.area << '\n';
    //measure views
    double pi = 2 * acos(0.0);
    double gold = 3 - sqrt(5);
    int samples = 50;
    for (int i = 0; i < samples; i++) {
        double z = 1 - (double(i) / double(samples - 1)) * 2;
        double theta = pi * i * gold;
        double x = cos(theta) * sqrt(1 - z * z);
        double y = sin(theta) * sqrt(1 - z * z);
        if (std::fabs(x) < 1e-5 && std::fabs(y) < 1e-5 && std::fabs(z) < 1e-5)continue;
        view_direction temp_view;
        temp_view.Dir = -K::Vector_3(x, y, z);
        measure_views.push_back(temp_view);
    }
    //int num_of_views = 35;
    //double const PI = 3.14159265;
    //for (int i = 1; i < num_of_views; ++i)
    //{
    //    auto phi = std::acos(-1.0 + (2.0 * i - 1.0) / num_of_views);
    //    auto theta = std::sqrt(num_of_views * PI) * phi;

    //    view_direction temp_view;
    //    auto dir = -K::Vector_3(std::cos(theta) * std::sin(phi),
    //        std::sin(theta) * std::sin(phi),
    //        std::cos(phi));
    //    temp_view.Dir = dir;
    //    measure_views.push_back(temp_view);
    //}


}
bool Runner::my_boolean_op(IK_Mesh& mesh1, IK_Mesh& mesh2, std::string op)
{
    if(!CGAL::is_closed(mesh1)|| !CGAL::is_closed(mesh2)||!PMP::does_bound_a_volume(mesh1)
        ||!PMP::does_bound_a_volume(mesh2))
        return false;
   /* Eigen::VectorXi J;
    auto libigl_mesh1 = *to_libigl_mesh(mesh1);
    auto libigl_mesh2 = *to_libigl_mesh(mesh2);

    if (op == "intersect")
    {
        igl::MeshBooleanType boolean_type(igl::MESH_BOOLEAN_TYPE_INTERSECT);
        igl::copyleft::cgal::mesh_boolean(libigl_mesh1.V, libigl_mesh1.F,
            libigl_mesh2.V, libigl_mesh2.F, boolean_type, libigl_mesh1.V, libigl_mesh1.F, J);
        mesh1 = *to_surface_mesh(libigl_mesh1);
        return true;
    }
    else if (op == "union")
    {
        igl::MeshBooleanType boolean_type(igl::MESH_BOOLEAN_TYPE_UNION);
        igl::copyleft::cgal::mesh_boolean(libigl_mesh1.V, libigl_mesh1.F,
            libigl_mesh2.V, libigl_mesh2.F, boolean_type, libigl_mesh1.V, libigl_mesh1.F, J);
        mesh1 = *to_surface_mesh(libigl_mesh1);
        return true;
    }
    else if (op == "substract")
    {
        igl::MeshBooleanType boolean_type(igl::MESH_BOOLEAN_TYPE_MINUS);
        igl::copyleft::cgal::mesh_boolean(libigl_mesh1.V, libigl_mesh1.F,
            libigl_mesh2.V, libigl_mesh2.F, boolean_type, libigl_mesh1.V, libigl_mesh1.F, J);
        mesh1 = *to_surface_mesh(libigl_mesh1);
        return true;
    }*/
    return false;
}
void Runner::init_origin_measure(Shader& shader)
{
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    glViewport(0, 0, measure_w, measure_w);
    for (int i = 0; i < measure_views.size(); ++i)
    {
        glGenFramebuffers(1, &measure_frame_buffer[i]);
        glBindFramebuffer(GL_FRAMEBUFFER, measure_frame_buffer[i]);
        unsigned int textureColorbuffer;
        glGenTextures(1, &textureColorbuffer);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)measure_w, (GLsizei)measure_w, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
        // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
        unsigned int rbo;
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,(GLsizei)measure_w, (GLsizei)measure_w); // use a single renderbuffer object for both a depth AND stencil buffer.
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
        // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";

        auto temp_dir = measure_views[i].Dir;
        auto unit_temp_dir = (Vec3(
            CGAL::to_double(temp_dir.x())
            , CGAL::to_double(temp_dir.y())
            , CGAL::to_double(temp_dir.z())));

        auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;

        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        
        shader.use();
        shader.setInteger("u", 1);
        
        glm::mat4 projection = glm::ortho(-1.5f, 1.5f, -1.5f, 1.5f, -1.0f, 100.0f);
        glm::mat4 view = glm::lookAt(temp_eye, Vec3(0,0,0), Vec3(0.0f,1.0f,0.0f));
        glm::mat4 model = glm::mat4(1);


        shader.setMat4("model", model);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);

        glEnable(GL_DEPTH_TEST);
        glClearColor(-1, -1, -1, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(tri_VAO);
        glDrawElements(GL_TRIANGLES, render_mesh.idxs.size(), GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDeleteRenderbuffers(1, &rbo);
        glDeleteTextures(1, &textureColorbuffer);
    }
    glViewport(0, 0, w, h);

}

void Runner:: render_for_primitives()
{
    auto start = clock();
    unsigned int primitive_rb,position_rb;
    EK_to_IK to_ik;
    for (int i = 0; i < 2; ++i)
    {
        glGenFramebuffers(1, &primitive_rb);
        glBindFramebuffer(GL_FRAMEBUFFER, primitive_rb);
        unsigned int textureColorbuffer;
        glGenTextures(1, &textureColorbuffer);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (float)128, (float)128, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
        // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
        unsigned int rbo;
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (float)128, (float)128); // use a single renderbuffer object for both a depth AND stencil buffer.
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
        // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";
        // get view matrix 
        auto temp_dir = Visual_hull_views[i].Dir;
        auto unit_temp_dir = glm::normalize(Vec3(
            CGAL::to_double(temp_dir.x())
            , CGAL::to_double(temp_dir.y())
            , CGAL::to_double(temp_dir.z())));
        auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)1280 / (float)1280, 0.1f, 100.0f);
        //glm::mat4 projection = glm::ortho(-2.f, 2.f, -2.f, 2.f, -2.0f, 2.0f);
        glm::mat4 view = glm::lookAt(temp_eye, Vec3(0, 0, 0), Vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = glm::mat4(1);
        glViewport(0, 0, 128, 128);

        shader.use();
        shader.setInteger("u", 1);
        shader.setMat4("model", model);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);

        glEnable(GL_DEPTH_TEST);
        glClearColor(-1, -1, -1, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(tri_VAO);
        glDrawElements(GL_TRIANGLES, render_mesh.idxs.size(), GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(0);

        //render lines
// if different color for point,edge,tris,use three different vao,vbo;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineWidth(1.5f);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        //glBindVertexArray(line_VAO);
        //glDrawElements(GL_TRIANGLES, (GLuint)render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(line_VAO);
        glDrawArrays(GL_LINES, 0, render_mesh.line_V_C_N.size() / 3);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glBindVertexArray(0);

        // render points

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPointSize(1.0);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        //glBindVertexArray(point_VAO);
        //glDrawElements(GL_TRIANGLES, render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(point_VAO);
        glDrawArrays(GL_POINTS, 0, render_mesh.point_V_C_N.size() / 3);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glBindVertexArray(0);

        //glGenFramebuffers(1, &position_rb);
        //glBindFramebuffer(GL_FRAMEBUFFER, position_rb);
        //unsigned int textureColorb;
        //glGenTextures(1, &textureColorb);
        //glBindTexture(GL_TEXTURE_2D, textureColorb);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (float)128, (float)128, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorb, 0);
        //// create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
        //unsigned int rb;
        //glGenRenderbuffers(1, &rb);
        //glBindRenderbuffer(GL_RENDERBUFFER, rb);
        //glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (float)128, (float)128); // use a single renderbuffer object for both a depth AND stencil buffer.
        //glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
        //// now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
        //if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        //    std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";
        //
        //glEnable(GL_DEPTH_TEST);
        //glClearColor(-1, -1, -1, 1);
        //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //map_shader.use();

        //glViewport(0, 0, 128, 128);
        //
        //map_shader.setMat4("model", model);
        //map_shader.setMat4("view", view);
        //map_shader.setMat4("projection", projection);

        //glBindVertexArray(map_VAO);
        //glDrawElements(GL_TRIANGLES, render_mesh.V_Idx.size() / 2, GL_UNSIGNED_INT, nullptr);
        //glDisable(GL_DEPTH_TEST);
        //glBindVertexArray(0);
        //glBindFramebuffer(GL_FRAMEBUFFER, 0);


        //---------------------------read to a matrix---------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, primitive_rb);
        glReadPixels(0, 0, 128, 128, GL_RGB, GL_FLOAT, matrix);
        //---------------------------bfs to find outer_bound--------------------
        memset(found, 0, sizeof(bool) * 128 * 128);
        int count_ = 0;
        //for (int x = 0; x < 128; ++x)
        //{
        //    for (int y = 0; y < 128; ++y)
        //    {
        //        for (int id = 0; id < 3; ++id)
        //        {
        //            if (matrix[(x * 128 + y) * 3 + id] > 1e-5)
        //            {
        //                for (int k = 0; k < 8; ++k)
        //                {
        //                    int nx = x + dx[k];
        //                    int ny = y + dy[k];
        //                    if (nx < 0 || ny < 0 || nx >= 128 || ny >= 128)continue;
        //                    if (matrix[(nx * 128 + ny) * 3] < 1e-5 && matrix[(nx * 128 + ny) * 3 + 1] < 1e-5 &&
        //                        matrix[(nx * 128 + ny) * 3 + 2] < 1e-5)
        //                    {
        //                        std::cout << x << " " << y << '\n';
        //                        break;
        //                    }
        //                }

        //            }
        //        }
        //    }
        //}
        // test how many bound points
        // 
        int count = 0;
        for (int x = 0; x < 128; ++x)
        {
            for (int y = 0; y < 128; ++y)
            {
                std::cout << matrix[(x * 128 + y) * 3]<<" ";
            }
        }

        // 
        // to find bound in 2D
        for (int x = 0; x < 128; ++x)
        {
            for (int y = 0; y < 128; ++y)
            {
                if (found[x][y])continue;
                bool need_to_traverse = false;
                for (int id = 0; id < 3; ++id)
                {
                    if (matrix[(x * 128 + y) * 3 + id] > 1e-5)
                    {
                        for (int k = 0; k < 8; ++k)
                        {
                            int nx = x + dx[k];
                            int ny = y + dy[k];
                            if (nx < 0 || ny < 0 || nx >= 128 || ny >= 128)continue;
                            if (matrix[(nx * 128 + ny) * 3] < 1e-5 && matrix[(nx * 128 + ny) * 3 + 1] < 1e-5 &&
                                matrix[(nx * 128 + ny) * 3 + 2] < 1e-5)
                            {
                                need_to_traverse = true;
                            }
                        }

                    }
                    if (need_to_traverse)break;
                }
                if (need_to_traverse)
                {
                    std::queue<PII> q;
                    std::vector<PII> bound;
                    std::vector<IK::Point_3> _3d_bound;
                    std::vector<IK::Point_3> decreased_3d_bound;
                    q.push({ x,y });
                    found[x][y] = true;
                    while (!q.empty())
                    {
                        auto t = q.front();
                        q.pop();
                        int tx = t.first, ty = t.second;
                        bool is_bound = false;
                        for (int k = 0; k < 8; ++k)
                        {
                            int nx = tx + dx[k];
                            int ny = ty + dy[k];
                            if (nx < 0 || ny < 0 || nx >= 128 || ny >= 128)continue;
                            if (matrix[(nx * 128 + ny) * 3] < 1e-5 && matrix[(nx * 128 + ny) * 3 + 1] < 1e-5 &&
                                matrix[(nx * 128 + ny) * 3 + 2] < 1e-5)is_bound = true;
                        }
                        if (!is_bound) continue;
                        //std::cout << tx << " " << ty << '\n';
                        //puts("find a bound");
                        bound.push_back({ tx,ty });
                        for (int k = 0; k < 8; ++k)
                        {
                            int nx = tx + dx[k];
                            int ny = ty + dy[k];
                            if (nx < 0 || ny < 0 || nx >= 128 || ny >= 128)continue;
                            if (found[nx][ny])continue;
                            if (matrix[(nx * 128 + ny) * 3] > 1e-5 || matrix[(nx * 128 + ny) * 3 + 1] > 1e-5 ||
                                matrix[(nx * 128 + ny) * 3 + 2] > 1e-5)
                            {
                                q.push({ nx,ny });
                                found[nx][ny] = true;
                            }
                        }

                    }
                    //transform bound to 3d bound!
                    for (auto& pos : bound)
                    {

                        auto projection_plane = to_ik(plane_3(Point_3(0, 0, 5), Visual_hull_views[i].Dir));
                        auto unit_plane = get_unit_plane(projection_plane);

                        float x = 2.0f * (float)pos.first / 128 - 1.0f;
                        float y = 1.0f - (2.0f * (float)pos.second / 128) ;//未反转
                        Vec3 ray_nds = Vec3(x, y, 1.0f);
                        glm::vec4 ray_clip = glm::vec4(ray_nds, 1.0f);

                        auto world_point = glm::inverse(view) * glm::inverse(projection) * ray_clip;
                        
                        if (std::fabs(world_point.w)>1e-5)
                        {
                            world_point.x = world_point.x / world_point.w;
                            world_point.y = world_point.y / world_point.w;
                            world_point.z = world_point.z / world_point.w;
                        }

                        glm::vec3 w_p = glm::vec3(world_point.x, world_point.y, world_point.z);

                        auto camera_pos = temp_eye;

                        IK::Point_3 p1(w_p.x, w_p.y, w_p.z);
                        IK::Point_3 p2(camera_pos.x, camera_pos.y, camera_pos.z);
                        IK::Line_3 ray(p1, p2);
                        
                        if (auto result =
                            CGAL::intersection(ray, unit_plane))
                        {
                            auto* inter_p = boost::get<IK::Point_3>(&*result);
                            _3d_bound.push_back(*inter_p);
                        }
                        else puts("false");

                    }    
                    decreased_3d_bound.assign(_3d_bound.begin(), _3d_bound.end());
                    //------------------------------------------decrease points in _3d_bound------------------------------------
                    /*for (int k = 0; k < _3d_bound.size(); )
                    {
                        if (k + 1 >= _3d_bound.size())decreased_3d_bound.push_back(_3d_bound[k]);
                        auto line = IK::Line_3(_3d_bound[k], _3d_bound[k + 1]);
                        int l = k + 2;
                        for (l; l < _3d_bound.size(); )
                        {
                            auto p_3 = _3d_bound[l];
                            if (CGAL::squared_distance(p_3, line) < 0.005)
                                ++l;
                            else break;
                        }
                        k = l;
                        decreased_3d_bound.push_back(_3d_bound[k]);
                        decreased_3d_bound.push_back(_3d_bound[l - 1]);
                    }*/
                    //decreased_3d_bound.assign(_3d_bound.begin(), _3d_bound.end());
                    std::cout << decreased_3d_bound.size() << "points_num\n";
                    //------------------------------------------get_primitives_____________________________________-------
                    if (decreased_3d_bound.size()<3)continue;
                    auto projection_plane = to_ik(plane_3(Point_3(0, 0, 5), Visual_hull_views[i].Dir));
                    auto unit_plane = get_unit_plane(projection_plane);
                    IK_Mesh solid_mesh;

                    std::vector<CGAL::SM_Vertex_index> p_side_vertices;
                    std::vector<CGAL::SM_Vertex_index> n_side_vertices;
                    for (auto p : decreased_3d_bound)
                    {
                        auto d = unit_plane.orthogonal_vector() * 15;
                        auto ek_p3 = p;
                        auto p_3_1 = ek_p3 + d;
                        auto p_3_2 = ek_p3 - d;
                        p_side_vertices.push_back(solid_mesh.add_vertex(p_3_1));
                        n_side_vertices.push_back(solid_mesh.add_vertex(p_3_2));
                    }

                    for (int j = 0; j < p_side_vertices.size(); ++j)
                    {
                        if (j != p_side_vertices.size() - 1) {
                            solid_mesh.add_face(n_side_vertices[j + 1], p_side_vertices[j + 1],
                                p_side_vertices[j]);
                            solid_mesh.add_face(p_side_vertices[j], n_side_vertices[j],
                                n_side_vertices[j+ 1]);
                        }
                        else {
                            solid_mesh.add_face(n_side_vertices[0], p_side_vertices[0],
                                p_side_vertices[j]);
                            solid_mesh.add_face(
                                p_side_vertices[j], n_side_vertices[j], n_side_vertices[0]);
                        }
                    }
                    std::vector<IK::Point_3> polyline;
                    for (int j = 0; j < p_side_vertices.size(); ++j)
                    {
                        polyline.push_back(solid_mesh.point(p_side_vertices[j]));
                    }
                    typedef CGAL::Triple<int, int, int> Triangle_int;
                    std::vector<Triangle_int> patch;
                    patch.reserve(polyline.size() - 2);
                    CGAL::Polygon_mesh_processing::triangulate_hole_polyline(
                        polyline,
                        std::back_inserter(patch));
                    for (std::size_t j = 0; j < patch.size(); ++j)
                    {
                        if (solid_mesh.add_face(p_side_vertices[patch[j].first],
                            p_side_vertices[patch[j].second],
                            p_side_vertices[patch[j].third]) == Mesh::null_face()) {
                            solid_mesh.add_face(p_side_vertices[patch[j].third],
                                p_side_vertices[patch[j].second],
                                p_side_vertices[patch[j].first]);
                        };
                    }
                    polyline.clear(); patch.clear();
                    for (int j = 0; j < n_side_vertices.size(); ++j)
                    {
                        polyline.push_back(solid_mesh.point(n_side_vertices[j]));
                    }
                    patch.reserve(polyline.size() - 2);
                    CGAL::Polygon_mesh_processing::triangulate_hole_polyline(
                        polyline,
                        std::back_inserter(patch));
                    for (std::size_t j = 0; j< patch.size(); ++j)
                    {
                        if (solid_mesh.add_face(n_side_vertices[patch[j].first],
                            n_side_vertices[patch[j].second],
                            n_side_vertices[patch[j].third]) == Mesh::null_face()) {
                            solid_mesh.add_face(n_side_vertices[patch[j].third],
                                n_side_vertices[patch[j].second],
                                n_side_vertices[patch[j].first]);
                        };
                    }
                    render_primitives.push_back(std::make_unique<IK_Mesh>(solid_mesh));
                    ++count_;
                }
            }
        }
        std::cout << count_ << "solid_mesh_found\n";
    }
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    auto end = clock();
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Total time:" << endtime << '\n';		//s为单位
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


//double Runner::difference_metric(Mesh& new_mesh,Shader& shader,bool use_boolean_type)
//{
//    double res = 0;
//    double pixels_num = measure_w * measure_w;
//    puts("yes");
//    transform(new_mesh, render_mesh_V);
//
//    glBindVertexArray(mtri_VAO);
//    glBindBuffer(GL_ARRAY_BUFFER, mtri_VBO);
//    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh_V.tri_V_C_N.size(), render_mesh_V.tri_V_C_N.data(), GL_DYNAMIC_DRAW);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mtri_EBO);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh_V.idxs.size(), render_mesh_V.idxs.data(), GL_DYNAMIC_DRAW);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
//    glEnableVertexAttribArray(2);
//
//    glBindVertexArray(mline_VAO);
//
//    glBindBuffer(GL_ARRAY_BUFFER, mline_VBO);
//    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh_V.line_V_C_N.size(), render_mesh_V.line_V_C_N.data(), GL_DYNAMIC_DRAW);
//
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mline_EBO);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh_V.idxs.size(), render_mesh_V.idxs.data(), GL_DYNAMIC_DRAW);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
//    glEnableVertexAttribArray(2);
//
//    glBindVertexArray(mpoint_VAO);
//
//    glBindBuffer(GL_ARRAY_BUFFER, mpoint_VBO);
//    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh_V.point_V_C_N.size(), render_mesh_V.point_V_C_N.data(), GL_DYNAMIC_DRAW);
//
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mpoint_EBO);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh_V.idxs.size(), render_mesh_V.idxs.data(), GL_DYNAMIC_DRAW);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
//    glEnableVertexAttribArray(2);
//    int w, h;
//    glfwGetWindowSize(window, &w, &h);
//    glViewport(0, 0, measure_w, measure_w);
//    int count = 0;
//    for (int i = 0; i < measure_views.size(); ++i)
//    {
//        double ans = 0.0;
//        
//        auto temp_dir = measure_views[i].Dir;
//        auto unit_temp_dir = Vec3(
//            CGAL::to_double(temp_dir.x())
//            , CGAL::to_double(temp_dir.y())
//            , CGAL::to_double(temp_dir.z()));
//
//        auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;
//
//        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
//
//        shader.use();
//        shader.setInteger("u", 1);
//
//        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)1 / (float)1, 0.1f, 100.0f);
//        glm::mat4 view = glm::lookAt(temp_eye, Vec3(0, 0, 0), Vec3(0.0f,1.0f,0.0f));
//        glm::mat4 model = glm::mat4(1);
//
//        shader.setMat4("model", model);
//        shader.setMat4("view", view);
//        shader.setMat4("projection", projection);
//        glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
//        glEnable(GL_DEPTH_TEST);
//        glClearColor(-1, -1, -1, 1);
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//        glBindVertexArray(mtri_VAO);
//        glDrawElements(GL_TRIANGLES, render_mesh_V.idxs.size(), GL_UNSIGNED_INT, 0);
//        glDisable(GL_DEPTH_TEST);
//        glBindVertexArray(0);
//
//        //render lines
//// if different color for point,edge,tris,use three different vao,vbo;
//        glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//        glBindFramebuffer(GL_FRAMEBUFFER, measure_frame_buffer[i]);
//        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_i);
//        glBindFramebuffer(GL_FRAMEBUFFER,measure_framebuffer_new_mesh);
//        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_v);
//        for (int k = 0; k < measure_w; ++k)
//        {
//            for (int j = 0; j < measure_w; ++j)
//            {
//                for (int id = 0; id < 3; ++id)
//                {
//                    if (!use_boolean_type)
//                        ans += (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id])
//                        * (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id]);
//                    else ans += ((m_i[(k * measure_w + j) * 3 + id] > 1e-5 && m_v[(k * measure_w + j) * 3 + id] < 1e-5) ||
//                        ((m_i[(k * measure_w + j) * 3 + id] < 1e-5 && m_v[(k * measure_w + j) * 3 + id] > 1e-5)));
//                }
//                    
//            }
//        }
//        ans /= pixels_num;
//        res += ans;;
//    }
//    glViewport(0, 0, w, h);
//    return res;
//}

double Runner::difference_metric(IK_Mesh& new_mesh, Shader& shader, bool use_boolean_type)
{
    puts("begin");
    glGenFramebuffers(1, &measure_framebuffer_new_mesh);
    glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
    unsigned int textureColorbuffer;
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, measure_w, measure_w, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, measure_w, measure_w); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";
    double res = 0;
    //puts("transform");
    transform_ik(new_mesh, render_mesh_V);
    //puts("transform finish");

    glBindVertexArray(mtri_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, mtri_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh_V.tri_V_C_N.size(), render_mesh_V.tri_V_C_N.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mtri_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh_V.idxs.size(), render_mesh_V.idxs.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
    glEnableVertexAttribArray(2);

    int count = 0;
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    glViewport(0, 0, measure_w, measure_w);
    for (int i = 0; i < measure_views.size(); ++i)
    {
        double ans = 0.0;

        auto temp_dir = measure_views[i].Dir;
        auto unit_temp_dir = Vec3(
            CGAL::to_double(temp_dir.x())
            , CGAL::to_double(temp_dir.y())
            , CGAL::to_double(temp_dir.z()));
        auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;

        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);


        shader.use();
        shader.setInteger("u", 1);

        glm::mat4 projection = glm::ortho(-1.5f, 1.5f, -1.5f, 1.5f, -1.0f, 100.0f);        
        glm::mat4 view = glm::lookAt(temp_eye, Vec3(0, 0, 0), Vec3(0.0f,1.0f,0.0f));
        glm::mat4 model = glm::mat4(1);

        shader.setMat4("model", model);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
        glEnable(GL_DEPTH_TEST);
        glClearColor(-1, -1, -1, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(mtri_VAO);
        glDrawElements(GL_TRIANGLES, render_mesh_V.idxs.size(), GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(0);

        //render lines
// if different color for point,edge,tris,use three different vao,vbo;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, measure_frame_buffer[i]);
        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_i);
        glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_v);
        for (int k = 0; k < measure_w; ++k)
        {
            for (int j = 0; j < measure_w; ++j)
            {
                for (int id = 0; id < 3; ++id)
                {
                    if (!use_boolean_type)
                        ans += (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id])
                        * (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id]);
                    else {
                        if ((m_i[(k * measure_w + j) * 3 + id] > 1e-5 && m_v[(k * measure_w + j) * 3 + id] < 1e-5) ||
                            ((m_i[(k * measure_w + j) * 3 + id] < 1e-5 && m_v[(k * measure_w + j) * 3 + id] > 1e-5)))
                            ans += 1.0;
                    }
                }

            }
        }
        res = std::max(res, ans);
    }
    glViewport(0, 0, w, h);
    glDeleteFramebuffers(1, &measure_framebuffer_new_mesh);
    glDeleteTextures(1, &textureColorbuffer);
    glDeleteRenderbuffers(1, &rbo);
    puts("end");
    return res;
}
double Runner::difference_metric_polyhedron(Polyhedron& new_mesh, Shader& shader, bool use_boolean_type)
{
    glGenFramebuffers(1, &measure_framebuffer_new_mesh);
    glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
    unsigned int textureColorbuffer;
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, measure_w, measure_w, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, measure_w, measure_w); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";
    double res = 0;
    //puts("transform");
    transform_polyhedron(new_mesh, render_mesh_V);
    //puts("transform finish");

    glBindVertexArray(mtri_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, mtri_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh_V.tri_V_C_N.size(), render_mesh_V.tri_V_C_N.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mtri_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh_V.idxs.size(), render_mesh_V.idxs.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
    glEnableVertexAttribArray(2);

    int count = 0;
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    glViewport(0, 0, measure_w, measure_w);
    for (int i = 0; i < measure_views.size(); ++i)
    {
        double ans = 0.0;

        auto temp_dir = measure_views[i].Dir;
        auto unit_temp_dir = Vec3(
            CGAL::to_double(temp_dir.x())
            , CGAL::to_double(temp_dir.y())
            , CGAL::to_double(temp_dir.z()));
        auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;

        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);


        shader.use();
        shader.setInteger("u", 1);

        glm::mat4 projection = glm::ortho(-1.5f, 1.5f, -1.5f, 1.5f, -1.0f, 100.0f);
        glm::mat4 view = glm::lookAt(temp_eye, Vec3(0, 0, 0), Vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = glm::mat4(1);

        shader.setMat4("model", model);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
        glEnable(GL_DEPTH_TEST);
        glClearColor(-1, -1, -1, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(mtri_VAO);
        glDrawElements(GL_TRIANGLES, render_mesh_V.idxs.size(), GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(0);

        //render lines
// if different color for point,edge,tris,use three different vao,vbo;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, measure_frame_buffer[i]);
        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_i);
        glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_v);
        for (int k = 0; k < measure_w; ++k)
        {
            for (int j = 0; j < measure_w; ++j)
            {
                for (int id = 0; id < 3; ++id)
                {
                    if (!use_boolean_type)
                        ans += (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id])
                        * (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id]);
                    else {
                        if ((m_i[(k * measure_w + j) * 3 + id] > 1e-5 && m_v[(k * measure_w + j) * 3 + id] < 1e-5) ||
                            ((m_i[(k * measure_w + j) * 3 + id] < 1e-5 && m_v[(k * measure_w + j) * 3 + id] > 1e-5)))
                            ans += 1.0;
                    }
                }

            }
        }
        res = std::max(res, ans);
    }
    glViewport(0, 0, w, h);
    glDeleteFramebuffers(1, &measure_framebuffer_new_mesh);
    glDeleteTextures(1, &textureColorbuffer);
    glDeleteRenderbuffers(1, &rbo);
    return res;
}
double Runner::difference_metric(libgl_mesh& new_mesh, Shader& shader, bool use_boolean_type)
{
    glGenFramebuffers(1, &measure_framebuffer_new_mesh);
    glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
    unsigned int textureColorbuffer;
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, measure_w, measure_w, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, measure_w, measure_w); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";
    double res = 0;
    //puts("transform");
    transform(new_mesh, render_mesh_V);
    //puts("transform finish");

    glBindVertexArray(mtri_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, mtri_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh_V.tri_V_C_N.size(), render_mesh_V.tri_V_C_N.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mtri_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh_V.idxs.size(), render_mesh_V.idxs.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
    glEnableVertexAttribArray(2);

    int count = 0;
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    glViewport(0, 0, measure_w, measure_w);
    for (int i = 0; i < measure_views.size(); ++i)
    {
        double ans = 0.0;

        auto temp_dir = measure_views[i].Dir;
        auto unit_temp_dir = Vec3(
            CGAL::to_double(temp_dir.x())
            , CGAL::to_double(temp_dir.y())
            , CGAL::to_double(temp_dir.z()));
        auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;

        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);


        shader.use();
        shader.setInteger("u", 1);

        glm::mat4 projection = glm::ortho(-1.5f, 1.5f, -1.5f, 1.5f, -1.0f, 100.0f);
        glm::mat4 view = glm::lookAt(temp_eye, Vec3(0, 0, 0), Vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = glm::mat4(1);

        shader.setMat4("model", model);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
        glEnable(GL_DEPTH_TEST);
        glClearColor(-1, -1, -1, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(mtri_VAO);
        glDrawElements(GL_TRIANGLES, render_mesh_V.idxs.size(), GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(0);

        //render lines
// if different color for point,edge,tris,use three different vao,vbo;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, measure_frame_buffer[i]);
        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_i);
        glBindFramebuffer(GL_FRAMEBUFFER, measure_framebuffer_new_mesh);
        glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_v);
        for (int k = 0; k < measure_w; ++k)
        {
            for (int j = 0; j < measure_w; ++j)
            {
                for (int id = 0; id < 3; ++id)
                {
                    if (!use_boolean_type)
                        ans += (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id])
                        * (double)(m_i[(k * measure_w + j) * 3 + id] - m_v[(k * measure_w + j) * 3 + id]);
                    else {
                        if ((m_i[(k * measure_w + j) * 3 + id] > 1e-5 && m_v[(k * measure_w + j) * 3 + id] < 1e-5) ||
                            ((m_i[(k * measure_w + j) * 3 + id] < 1e-5 && m_v[(k * measure_w + j) * 3 + id] > 1e-5)))
                            ans += 1.0;
                    }
                }

            }
        }
        res = std::max(res, ans);
    }
    glViewport(0, 0, w, h);
    glDeleteFramebuffers(1, &measure_framebuffer_new_mesh);
    glDeleteTextures(1, &textureColorbuffer);
    glDeleteRenderbuffers(1, &rbo);
    return res;
}

void Runner::set_M_v_bbox()
{
    IK_Mesh mesh_V;

    auto bounding_box = CGAL::bounding_box(ik_mesh.points().begin(), ik_mesh.points().end());
    auto xmin = bounding_box.xmin();
    auto xmax = bounding_box.xmax();
    auto ymin = bounding_box.ymin();
    auto ymax = bounding_box.ymax();
    auto zmin = bounding_box.zmin();
    auto zmax = bounding_box.zmax();

    auto p1 = mesh_V.add_vertex(IK::Point_3(xmin, ymin, zmax));
    auto p2 = mesh_V.add_vertex(IK::Point_3(xmax, ymin, zmax));
    auto p3 = mesh_V.add_vertex(IK::Point_3(xmax, ymin, zmin));
    auto p4 = mesh_V.add_vertex(IK::Point_3(xmin, ymin, zmin));
    auto p5 = mesh_V.add_vertex(IK::Point_3(xmin, ymax, zmax));
    auto p6 = mesh_V.add_vertex(IK::Point_3(xmax, ymax, zmax));
    auto p7 = mesh_V.add_vertex(IK::Point_3(xmax, ymax, zmin));
    auto p8 = mesh_V.add_vertex(IK::Point_3(xmin, ymax, zmin));
    
    mesh_V.add_face(p1, p4, p3);
    mesh_V.add_face(p3, p2, p1);
    mesh_V.add_face(p5, p6, p7);
    mesh_V.add_face( p7, p8,p5);
    mesh_V.add_face(p1, p2, p6);
    mesh_V.add_face( p6, p5,p1);
    mesh_V.add_face(p2, p3, p7);
    mesh_V.add_face( p7, p6,p2);
    mesh_V.add_face(p3, p4, p8);
    mesh_V.add_face( p8, p7,p3);
    mesh_V.add_face(p1, p5, p8);
    mesh_V.add_face( p8, p4,p1);
    const std::string path = SAVE_PATH;
    const std::string fullpath = path + "bbox" + ".off";
    std::ofstream out(fullpath);
    out << (mesh_V);
    std::ifstream ifs1(fullpath);
    Polyhedron p;
    ifs1 >> p;
    Visual_hull = Nef_polyhedron(p);
    mesh_Vs.push_back(std::make_unique<IK_Mesh>(mesh_V));
}
void Runner::get_carved_meshes()
{
    std::vector<Mesh_area> meshes;
#pragma omp parallel for 
    for (int z =0; z<Config::Path::get().planes; ++z)
    {
        auto R_p = carve_planes[z];
        IK::Plane_3 plane = R_p.plane;
        IK::Plane_3 unit_plane = plane;
        auto t_point = unit_plane.point();
        auto othr = unit_plane.orthogonal_vector();
        if (!unit_plane.has_on_positive_side(t_point + othr))othr = -othr;
        auto add_sub = 3 * othr;
        std::vector<Polygon_2> polygons_positive;
        //std::vector<Polygon_2> polygons_negative;
        std::vector<K::Point_2> box_points_p;
        for (auto& face:ik_mesh.faces())
        {
            if (PMP::is_degenerate_triangle_face(face, ik_mesh))continue;
            if (R_p.fds_set.count(face))continue;
            Polygon_2 poly;
            std::vector<IK::Point_2> after_project_points;
            bool is_positive_side = false;
            //bool is_negative_side = false;
            for (auto vertex : ik_mesh.vertices_around_face(ik_mesh.halfedge(face)))
            {
                auto point = ik_mesh.point(vertex);
                if (unit_plane.has_on_positive_side(point))is_positive_side = true;
                IK::Point_3 pro_point = unit_plane.projection(point);
                after_project_points.push_back(unit_plane.to_2d(pro_point));//to_2d !
            }
            IK::Triangle_2 tri(after_project_points[0], after_project_points[1], after_project_points[2]);
            if (tri.is_degenerate())
                continue;
            //assert(after_project_points.size() == 3);
            if (!is_positive_side)continue;
            
            for (auto p : after_project_points)
            {
                poly.push_back(p);
                box_points_p.push_back(p);
            }
            if (!poly.is_counterclockwise_oriented())
                poly.reverse_orientation();
            polygons_positive.push_back(poly);
        }
        if (polygons_positive.size() == 0)continue;
        std::vector<Polygon_with_holes_2> p_w_h_positive;
        CGAL::join(polygons_positive.begin(), polygons_positive.end(), std::back_inserter(p_w_h_positive));
        CGAL::Bbox_2 box_p;
        
        if (polygons_positive.size())
        {
            auto b_p = CGAL::bounding_box(box_points_p.begin(), box_points_p.end());
            box_p = b_p.bbox();
            double length = std::fmax((box_p.xmax() - box_p.xmin()), (box_p.ymax() - box_p.ymin()));
            Polygon_2 p_box_poly;
            K::Point_2 p1(box_p.xmin() - 1, box_p.ymin() -1),
                p2(box_p.xmax()+1 , box_p.ymin() -1)
                , p3(box_p.xmax() + 1, box_p.ymax() + 1)
                , p4(box_p.xmin() -1, box_p.ymax() +1);
            p_box_poly.push_back(p1), p_box_poly.push_back(p2), p_box_poly.push_back(p3), p_box_poly.push_back(p4);

            IK_Mesh all_outer_boundary_mesh;
            for (auto& pwh : p_w_h_positive)// holes is carve mesh
            {
                auto out_boundary = pwh.outer_boundary();
                if (out_boundary.area() < 0.001)continue;
                PS::Squared_distance_cost cost;
                PS::Stop_above_cost_threshold stop(length / 10000);
                out_boundary = PS::simplify(out_boundary, cost, stop);

                auto out_boundary_mesh = *from_poly_to_mesh(out_boundary, unit_plane, add_sub, false);
                all_outer_boundary_mesh += out_boundary_mesh;
                for (auto& hole : pwh.holes())
                {
                    if (hole.area() < 0.0001)continue;
                    PS::Squared_distance_cost cost;
                    PS::Stop_above_cost_threshold stop(length / 10000);
                    hole = PS::simplify(hole, cost, stop);
                    auto carve_mesh = *from_poly_to_mesh(hole, unit_plane, add_sub, false);
#pragma omp critical
                    {
                        if (CGAL::is_closed(carve_mesh))
                        {
                            auto mesh = Mesh_area(carve_mesh, R_p.area);
                            meshes.push_back(mesh);
                        }
                    }
                }
            }
            auto bbox_mesh = *from_poly_to_mesh(p_box_poly, unit_plane, add_sub, false);
            bool box_intersecting = PMP::does_self_intersect<CGAL::Parallel_if_available_tag>(bbox_mesh, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, bbox_mesh)));
            bool boundary_intersecting = PMP::does_self_intersect<CGAL::Parallel_if_available_tag>(all_outer_boundary_mesh, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, all_outer_boundary_mesh)));

            if (!box_intersecting && !boundary_intersecting)
            {
                IK_Mesh out_mesh;
                PMP::corefine_and_compute_difference(bbox_mesh, all_outer_boundary_mesh, out_mesh);

#pragma omp critical
                {
                    if (CGAL::is_closed(out_mesh))
                    {
                        auto mesh = Mesh_area(out_mesh, R_p.area);
                        meshes.push_back(mesh);
                    }
                }
            }
        }
    }
    std::sort(meshes.begin(), meshes.end());
    for (auto& mesh : meshes)
    {
        carve_meshes.push_back(std::make_unique<IK_Mesh>(mesh.mesh));
    }

}
void Runner::get_primitives()
{
    //glGenFramebuffers(1, &View_buffer);
    //glBindFramebuffer(GL_FRAMEBUFFER, View_buffer);
    //unsigned int textureColorbuffer;
    //glGenTextures(1, &textureColorbuffer);
    //glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)measure_w, (GLsizei)measure_w, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
    //// create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    //unsigned int rbo;
    //glGenRenderbuffers(1, &rbo);
    //glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    //glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (GLsizei)measure_w, (GLsizei)measure_w); // use a single renderbuffer object for both a depth AND stencil buffer.
    //glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
    //// now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
    //if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    //    std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";

    //std::cout << Visual_hull_views.size();
    //int w, h;
    //glfwGetWindowSize(window, &w, &h);
    //glViewport(0, 0, measure_w, measure_w);
    //for (int i = 0; i < Visual_hull_views.size(); ++i)
    //{
    //    std::unordered_set<int> View_f_set;
    //    glBindFramebuffer(GL_FRAMEBUFFER, View_buffer);
    //    auto temp_dir = Visual_hull_views[i].Dir;
    //    auto unit_temp_dir = (Vec3(
    //        CGAL::to_double(temp_dir.x())
    //        , CGAL::to_double(temp_dir.y())
    //        , CGAL::to_double(temp_dir.z())));
    //    auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;
    //    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    //    //glm::mat4 projection = glm::perspective(glm::radians(fov), (float)measure_w / (float)measure_w, 0.1f, 100.0f);
    //    glm::mat4 projection = glm::ortho(-2.f, 2.f, -2.f, 2.f, -2.0f, 100.0f);
    //    glm::mat4 view = glm::lookAt(temp_eye, Vec3(0.f,0.f,0.f),Vec3(0.0f,1.0f,0.0f));
    //    glm::mat4 model = glm::mat4(1);
    //    glEnable(GL_DEPTH_TEST);
    //    glClearColor(-1, -1, -1, 1);
    //    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //    map_shader.use();
    //    map_shader.setMat4("model", model);
    //    map_shader.setMat4("view", view);
    //    map_shader.setMat4("projection", projection);
    //    glPointSize(10.0);
    //    glBindVertexArray(map_VAO);
    //    glDrawElements(GL_TRIANGLES, render_mesh.V_Idx.size() / 2, GL_UNSIGNED_INT, nullptr);
    //    glDisable(GL_DEPTH_TEST);
    //    glBindVertexArray(0);
    //    glReadPixels(0, 0, measure_w, measure_w, GL_RGB, GL_FLOAT, m_i);
    //    for (int x = 0; x < measure_w; ++x)
    //    {
    //        for (int y = 0; y < measure_w; ++y)
    //        {
    //            if (m_i[(x * measure_w + y) * 3] < 0)continue;
    //            unsigned int idx = reverse_transform_idx(Vec3(m_i[(x * measure_w + y) * 3],
    //                m_i[(x * measure_w + y) * 3+1], m_i[(x * measure_w + y) * 3+2]));
    //            if (idx <= 0)continue;
    //            View_f_set.insert(idx-1);
    //        }
    //    }
    //    faces_can_see.push_back(View_f_set);
    //    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //}
    //glViewport(0, 0, w, h);

//#pragma omp parallel for
//    for (int h = 0; h < V_hull_views.size(); ++h)
//    {
//        auto& poly_t = V_hull_views[h].pwh;
//        auto unit_plane = V_hull_views[h].plane;
//        PS::Squared_distance_cost cost;
//        PS::Stop_above_cost_threshold stop(0.0025);
//        poly_t = PS::simplify(poly_t, cost, stop);
//        auto bbox = poly_t.bbox();
//        //solid
//        IK_Mesh solid_mesh;
//        Polygon_2 out_poly = poly_t.outer_boundary();
//        if (!out_poly.is_counterclockwise_oriented())out_poly.reverse_orientation();
//        if (!out_poly.is_counterclockwise_oriented())std::cout << "not orientation\n";
//        //polygon_simplfy(out_poly);
//        if (out_poly.size() >= 3)
//        {
//            solid_mesh = *from_poly_to_mesh(out_poly, unit_plane, 5.0 * unit_plane.orthogonal_vector());
//#pragma omp critical
//            {
//                simplify(solid_mesh);
//            }
//            if (CGAL::is_closed(solid_mesh) && PMP::does_bound_a_volume(solid_mesh) && !PMP::does_self_intersect(solid_mesh))
//                primitives.push_back(std::make_unique<IK_Mesh>(solid_mesh));
//        }
//        //hollow
//        Polygon_2 bbox_poly;
//        bbox_poly.push_back(Point_2(bbox.xmin(), bbox.ymin()));
//        bbox_poly.push_back(Point_2(bbox.xmin(), bbox.ymax()));
//        bbox_poly.push_back(Point_2(bbox.xmax(), bbox.ymax()));
//        bbox_poly.push_back(Point_2(bbox.xmax(), bbox.ymin()));
//
//        for (auto& hollow : poly_t.holes())//
//        {
//            IK_Mesh hollow_mesh;
//            std::vector<Polygon_with_holes_2> pwhs;
//            CGAL::difference(bbox_poly, hollow, std::back_inserter(pwhs));
//            if (pwhs.size() != 1)std::cout << "why not just one pwh\n";
//            auto pwh = pwhs[0];
//            from_pwh_to_mesh(pwh, unit_plane, 5.0 * unit_plane.orthogonal_vector(), true);
//            if (CGAL::is_closed(hollow_mesh) && !PMP::does_self_intersect(hollow_mesh) && PMP::does_bound_a_volume(hollow_mesh))
//                primitives.push_back(std::make_unique<IK_Mesh>(hollow_mesh));
//        }
//    }

#pragma omp parallel
    {
    #pragma omp for nowait
        for (int h = 0; h < Visual_hull_views.size(); ++h)
        {
            //if (faces_can_see[h].size() == 0)continue;
            EK_to_IK to_ik;
            IK_to_EK to_ek;

            std::vector<Polygon_2> polygons;
            auto view = Visual_hull_views[h];

            auto projection_plane = to_ik(plane_3(Point_3(0, 0, 3), view.Dir));
            auto unit_plane = get_unit_plane(projection_plane);
            // try keep just one polygon_with_hole

            for (auto face : ik_mesh.faces())
            {
                //auto face = CGAL::SM_Face_index(f_idx);
                Polygon_2 poly;
                std::vector<IK::Point_2> after_project_points;
                for (auto vertex : ik_mesh.vertices_around_face(ik_mesh.halfedge(face)))
                {
                    auto point = ik_mesh.point(vertex);
                    IK::Point_3 pro_point = unit_plane.projection(point);
                    after_project_points.push_back(unit_plane.to_2d(pro_point));//to_2d !
                }
                IK::Triangle_2 tri(after_project_points[0], after_project_points[1], after_project_points[2]);
                if (tri.is_degenerate())
                    continue;
                //assert(after_project_points.size() == 3);
                for (auto p : after_project_points)
                    poly.push_back(to_ek(p));
                if (!poly.is_counterclockwise_oriented())
                    poly.reverse_orientation();

                polygons.push_back(poly);
            }

            std::vector<Polygon_with_holes_2> p_w_h;
            CGAL::join(polygons.begin(), polygons.end(), std::back_inserter(p_w_h));
            auto poly_t = p_w_h[0];
            for (int i = 1; i < p_w_h.size(); ++i)
                if (p_w_h[i].outer_boundary().size() > poly_t.outer_boundary().size())
                    poly_t = p_w_h[i];
            PS::Squared_distance_cost cost;
            PS::Stop_above_cost_threshold stop(0.0025);
            poly_t = PS::simplify(poly_t, cost, stop);
            auto bbox = poly_t.bbox();
            //solid
            IK_Mesh solid_mesh;
            Polygon_2 out_poly = poly_t.outer_boundary();
            if (!out_poly.is_counterclockwise_oriented())out_poly.reverse_orientation();
            if (!out_poly.is_counterclockwise_oriented())std::cout << "not orientation\n";
            //polygon_simplfy(out_poly);
            if (out_poly.size() >= 3)
            {
                solid_mesh = *from_poly_to_mesh(out_poly, unit_plane, 5.0 * unit_plane.orthogonal_vector(),true);
#pragma omp critical
                {
                    simplify(solid_mesh);
                }
                if (CGAL::is_closed(solid_mesh) && PMP::does_bound_a_volume(solid_mesh) && !PMP::does_self_intersect(solid_mesh))
                    primitives.push_back(std::make_unique<IK_Mesh>(solid_mesh));
            }
            //hollow
            Polygon_2 bbox_poly;
            bbox_poly.push_back(Point_2(bbox.xmin(), bbox.ymin()));
            bbox_poly.push_back(Point_2(bbox.xmin(), bbox.ymax()));
            bbox_poly.push_back(Point_2(bbox.xmax(), bbox.ymax()));
            bbox_poly.push_back(Point_2(bbox.xmax(), bbox.ymin()));

            for (auto& hollow : poly_t.holes())//
            {
                IK_Mesh hollow_mesh;
                std::vector<Polygon_with_holes_2> pwhs;
                CGAL::difference(bbox_poly, hollow, std::back_inserter(pwhs));
                if (pwhs.size() != 1)std::cout << "why not just one pwh\n";
                auto pwh = pwhs[0];
                from_pwh_to_mesh(pwh, unit_plane, 5.0 * unit_plane.orthogonal_vector(),true);
                if (CGAL::is_closed(hollow_mesh) && !PMP::does_self_intersect(hollow_mesh) && PMP::does_bound_a_volume(hollow_mesh))
                    primitives.push_back(std::make_unique<IK_Mesh>(hollow_mesh));
            }

        }
    }
}

void Runner::compute_M_v()
{
    return;
    Polyhedron pvh;
    Visual_hull.convert_to_polyhedron(pvh);
    PMP::triangulate_faces(pvh);
    double V_hull_metric = difference_metric_polyhedron(pvh, shader, true);
    for (auto& primitive : primitives_polyhedron)
    {
        const auto temp_V_h = Visual_hull - (Visual_hull - primitive);
        Polyhedron t_polyhedron;
        temp_V_h.convert_to_polyhedron(t_polyhedron);
        if (!CGAL::is_closed(t_polyhedron))continue;
        PMP::triangulate_faces(t_polyhedron);
        auto diff = difference_metric_polyhedron(t_polyhedron, shader, true);
        if (diff < V_hull_metric)
        {
            V_hull_metric = diff;
            Visual_hull = temp_V_h;
        }
    }
    /*Polyhedron pvh;
    Visual_hull.convert_to_polyhedron(pvh);
    PMP::triangulate_faces(pvh);
    double V_hull_metric = difference_metric_polyhedron(pvh, shader, true);
    double best_metric = V_hull_metric;
    const size_t size = primitives_polyhedron.size();
    bool *used = new bool[size]{false};
    Nef_polyhedron best_polyhedron = Visual_hull;
    std::cout << "begin\n";
    auto start = clock();
    while (true)
    {
        int best_primitives_id = 0;
        std::vector<Nef_polyhedron> temp_Vhs;
#pragma omp parallel for
        for (int i = 0;i<primitives_polyhedron.size();++i)
        {
            if (used[i])continue;
            const auto& temp = primitives_polyhedron[i];
            auto temp_Vh = Visual_hull - (Visual_hull - temp);
            temp_Vhs.push_back(temp_Vh);

        }
        for (int i = 0; i < temp_Vhs.size(); ++i)
        {
            auto& temp_Vh = temp_Vhs[i];
            Polyhedron t_polyhedron;
            temp_Vh.convert_to_polyhedron(t_polyhedron);
            PMP::triangulate_faces(t_polyhedron);
            auto diff = difference_metric_polyhedron(t_polyhedron, shader, true);
            if (diff < best_metric)
            {
                best_primitives_id = i;
                best_metric = diff;
                best_polyhedron = temp_Vh;
            }
        }
        if (best_metric < V_hull_metric - 500)
        {
            used[best_primitives_id] = true;
            V_hull_metric = best_metric;
            Visual_hull = best_polyhedron;
        }
        else break;
    }
    auto end = clock();
    std::cout << "Viusal_hull time:"<<end-start<<"\n";
    delete used;*/

}

//void Runner::compute_M_v()
//{
//    auto mesh_v = *mesh_Vs.back();
//    mesh_Vs.pop_back();
//    double best_difference = 1e9; 
//    int cnt = 0;
//    double stop_c = 800;
//    IK_Mesh best_mesh;
//    std::vector<IK_Mesh> primitives_mesh;
//    for (int i = 0; i < primitives.size(); ++i)primitives_mesh.push_back(*primitives[i]);
//    auto start = clock();
//
//    while (cnt < 10)
//    {
//        double this_loop_best = 1e9;
//        std::vector<IK_Mesh> loop_output;
//#pragma omp parallel for
//        for (int i = 0; i < primitives.size(); ++i)
//        {
//            auto cut_mesh = mesh_v;
//            IK_Mesh out_;
//            auto primitive = primitives_mesh[i];
//            if (!CGAL::is_closed(cut_mesh) || !CGAL::is_closed(primitive) ||!PMP::does_bound_a_volume(cut_mesh) || !PMP::does_bound_a_volume(primitive))
//                continue;
//            
//            PMP::corefine_and_compute_intersection(cut_mesh, primitive, out_);
//            //bool success_ = my_boolean_op(cut_mesh, primitive, "intersect");
//#pragma omp critical
//            {
//                loop_output.push_back(out_);
//            }
//        }
//        std::vector<IK_Mesh> selected_loop_output;
//#pragma omp parallel for
//        for (int i = 0; i < loop_output.size(); ++i)
//        {
//            auto &out_mesh = loop_output[i];
//            my_simplify(out_mesh);
//            out_mesh.collect_garbage();
//            
//            if (CGAL::is_closed(out_mesh) || PMP::does_bound_a_volume(out_mesh))
//                selected_loop_output.push_back(out_mesh);
//        }
//        for (int i = 0; i < selected_loop_output.size(); ++i)
//        {
//            auto &out_mesh = selected_loop_output[i];
//            auto diff = difference_metric(out_mesh, shader, true);
//            {
//                if (diff < this_loop_best)
//                {
//                    this_loop_best = diff;
//                    best_mesh = out_mesh;
//                }
//            }
//        }
//        //std::cout << this_loop_best <<'\n';
//        if (best_difference - this_loop_best < stop_c)break;
//        best_difference = this_loop_best;
//        
//        mesh_v = best_mesh;
//
//        const std::string path_ = "E:/jdit/normal_output";
//        const std::string fullpath_ = path_ + "visual_hull_si" + std::to_string(cnt) + ".off";
//        std::ofstream out_(fullpath_);
//        out_ << (mesh_v);
//
//        ++cnt;
//    }
//    auto end = clock();
//    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
//    std::cout << "Total time:" << endtime << '\n';
//    mesh_Vs.push_back(std::make_unique<IK_Mesh>(mesh_v));
//}

void Runner::carving()
{
    //auto start = clock();
    //puts("start carving");
    //for (int i = 0; i < Nef_carve_meshes.size(); ++i)
    //{
    //    Nef_polyhedron t = Visual_hull - Nef_carve_meshes[i];
    //    if (!t.is_simple())continue;
    //    Polyhedron polyhedron;
    //    Visual_hull.convert_to_polyhedron(polyhedron);
    //    const std::string path = SAVE_PATH;
    //    const std::string fullpath = path + "visual_carve_end_af_si" +std::to_string(i)+".off";
    //    std::ofstream out(fullpath);
    //    out << polyhedron;
    //    Visual_hull = t;
    //}
    int cnt = 0;
    double best_difference = 1e9;
    double stop_c = 50;
    Polyhedron loop_best_mesh;
    auto start = clock();
    puts("start carving");
    while (cnt < 50)
    {
        ++cnt;
        std::unordered_map<int, int> loop_idx_polyhedron_idx;
        IK_Mesh Visual_mesh;
        double this_loop_best = 1e9;
        int best_idx = -1;
        std::vector<Nef_polyhedron>polyhedron_to_select;
#pragma omp parallel for
        for (int k = 0; k < Nef_carve_meshes.size(); ++k)
        {
            Nef_polyhedron t = Visual_hull - Nef_carve_meshes[k]; 
            if (!t.is_simple())continue;
#pragma omp critical
            {
                polyhedron_to_select.push_back(t);
            }
        }
        std::vector<Polyhedron> loop_out;
#pragma omp parallel for
        for (int k = 0; k < polyhedron_to_select.size(); ++k)
        {
            Polyhedron pMesh;
            polyhedron_to_select[k].convert_to_polyhedron(pMesh);
            PMP::triangulate_faces(pMesh);
#pragma omp critical
            {
                loop_idx_polyhedron_idx[(int)loop_out.size()] = k;
                loop_out.push_back((pMesh));
            }
        }
        for (int k = 0; k < loop_out.size(); ++k)
        {
            auto& out_mesh = loop_out[k];
            if (!CGAL::is_closed(out_mesh))continue;
            auto diff = difference_metric_polyhedron(out_mesh, shader, false);
            std::cout << diff << '\n';
            if (diff < this_loop_best)
            {
                best_idx = k;
                this_loop_best = diff;
            }
        }
        if (best_difference - this_loop_best < stop_c||best_idx == -1)
        {
            break;
        }
           
        best_difference = this_loop_best;
        Visual_hull = polyhedron_to_select[loop_idx_polyhedron_idx[best_idx]];
        loop_best_mesh = loop_out[best_idx];
        const std::string path = SAVE_PATH;
        const std::string fullpath = path + "visual_carve_end_af_si" +std::to_string(cnt)+".off";
        std::ofstream out(fullpath);
        out << (loop_best_mesh);
        ++cnt;
    }

    Polyhedron Visual_hull_polyhedron;
    Visual_hull.convert_to_polyhedron(Visual_hull_polyhedron);
    double init_volume = CGAL::to_double(PMP::volume(Visual_hull_polyhedron));
    while (true)//decrease volume while keep look
    {
        int best_cutId = -1;
        std::unordered_map<int, int> loop_idx_polyhedron_idx;
        double best_volume = init_volume;
        std::vector<Nef_polyhedron>polyhedron_to_select;
#pragma omp parallel for
        for (int k = 0; k < Nef_carve_meshes.size(); ++k)
        {
            Nef_polyhedron t = Visual_hull - Nef_carve_meshes[k];
            if (!t.is_simple())continue;
#pragma omp critical
            {
                polyhedron_to_select.push_back(t);
            }
        }
        std::vector<Polyhedron> loop_out;
#pragma omp parallel for
        for (int k = 0; k < polyhedron_to_select.size(); ++k)
        {
            Polyhedron pMesh;
            polyhedron_to_select[k].convert_to_polyhedron(pMesh);
            PMP::triangulate_faces(pMesh);
#pragma omp critical
            {
                loop_idx_polyhedron_idx[(int)loop_out.size()] = k;
                loop_out.push_back((pMesh));
            }
        }

        for (int k = 0; k < loop_out.size(); ++k)
        {
            auto& pmesh = loop_out[k];
            auto diff = difference_metric_polyhedron(pmesh, shader, false);
            if (diff - best_difference < 50.0)
            {
                auto V = CGAL::to_double(PMP::volume(pmesh));
                if (V < best_volume)
                {
                    best_volume = V;
                    best_cutId = k;
                }
            }
        }
        if (init_volume - best_volume < init_volume / 100.0)break;
        init_volume = best_volume;
        Visual_hull = polyhedron_to_select[loop_idx_polyhedron_idx[best_cutId]];

        const std::string path = SAVE_PATH;
        const std::string fullpath = path + "visual_carve_end_af_si" + std::to_string(best_volume) + ".off";
        std::ofstream out(fullpath);
        out << (loop_out[best_cutId]);
    }

    auto end = clock();
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Total time:" << endtime << '\n';
}
IK::Point_3 Runner::get_P(unsigned int f,int x,int y)
{
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    float rgb_arry[3];
    glReadPixels(x, y, 1, 1, GL_RGB, GL_FLOAT, rgb_arry);

    if (rgb_arry[0] < 0)return NULL_POINT;
    unsigned int idx = reverse_transform_idx(Vec3(rgb_arry[0],
        rgb_arry[1], rgb_arry[2]));
    if (idx <= 0)return NULL_POINT;

    size_t f_id = idx - 1;
    auto temp = CGAL::SM_Face_index(f_id);
    CGAL::SM_Vertex_index selected_v;
    float min_dis = 1e5 + 10;
    for (auto v : mesh.vertices_around_face(mesh.halfedge(temp)))
    {

        Vec2 p = mvp_transform(Vec3(CGAL::to_double(mesh.point(v).x()), CGAL::to_double(mesh.point(v).y()),
            CGAL::to_double(mesh.point(v).z())));
        if (distance(p, Vec2(x, y)) < min_dis) {
            min_dis = distance(p, Vec2(x, y));
            selected_v = v;
        }
    }
    EK_to_IK to_ik;
    auto p= mesh.point(selected_v);
    return to_ik(p);
}

Polygon_set_2 Runner::divide_conquer_join(std::vector<Polygon_2> polygons,int l,int r)
{
    Polygon_set_2 ans;
    if (l >= r)
    {
        ans.insert(polygons[l]);
        return ans;
    }
    int mid =( l + r )>> 1;
    Polygon_set_2 polygon_set_l, polygon_set_r;
    #pragma omp parallel sections
    {
#pragma omp section
        polygon_set_l = divide_conquer_join(polygons, l, mid);
#pragma omp section
        polygon_set_r = divide_conquer_join(polygons, mid + 1, r);
    };
    polygon_set_l.join(polygon_set_r);
    return polygon_set_l;
}

IK::Plane_3 Runner::get_unit_plane(IK::Plane_3 plane)
{
    IK_to_IEK to_iek;
    IEK_to_IK to_ik;
    auto iek_plane = to_iek(plane);
    double length = 0.0;
    length = std::sqrt(iek_plane.a() * iek_plane.a() + iek_plane.b() * iek_plane.b() + iek_plane.c() * iek_plane.c());
    iek_plane = inexact_K::Plane_3(iek_plane.a() / length, iek_plane.b() / length, iek_plane.c() / length, iek_plane.d() / length);
    return to_ik(iek_plane);
}

void render_3d()
{
    if (my_runner->got_render_mesh)
    {
        my_runner->render_obj(my_runner->shader);
        my_runner->shader.use();
        my_runner->shader.setInteger("u", 1);
        glEnable(GL_DEPTH_TEST);
        //glm::mat4 projection = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.1f, 100.0f);
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)1280 / (float)720, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(camera.eye(), camera.dir(), camera.up());

        glm::mat4 model = glm::mat4(1);

        my_runner->shader.setMat4("model", model);
        my_runner->shader.setMat4("view", view);
        my_runner->shader.setMat4("projection", projection);
        glBindVertexArray(my_runner->tri_VAO);
        glDrawElements(GL_TRIANGLES, my_runner->render_mesh.idxs.size(), GL_UNSIGNED_INT, 0);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(0);

        //render lines
// if different color for point,edge,tris,use three different vao,vbo;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineWidth(1.5f);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        //glBindVertexArray(line_VAO);
        //glDrawElements(GL_TRIANGLES, (GLuint)render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(my_runner->line_VAO);
        glDrawArrays(GL_LINES, 0, my_runner->render_mesh.line_V_C_N.size() / 3);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glBindVertexArray(0);

        // render points

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPointSize(1.0);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        //glBindVertexArray(point_VAO);
        //glDrawElements(GL_TRIANGLES, render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
        //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(my_runner->point_VAO);
        glDrawArrays(GL_POINTS, 0, my_runner->render_mesh.point_V_C_N.size() / 3);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glBindVertexArray(0);

        // render for framebuffer map

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glEnable(GL_DEPTH_TEST);
        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glClearColor(-1, -1, -1, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        my_runner->map_shader.use();
        my_runner->map_shader.setMat4("model", model);
        my_runner->map_shader.setMat4("view", view);
        my_runner->map_shader.setMat4("projection", projection);
        glPointSize(10.0);
        glBindVertexArray(my_runner->map_VAO);
        glDrawElements(GL_TRIANGLES, my_runner->render_mesh.V_Idx.size() / 2, GL_UNSIGNED_INT, nullptr);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    glfwSwapBuffers(window);
}


void Runner::get_mesh(std::vector<Point_3>& points, std::vector<std::vector<unsigned int>> indices)
{
    const auto bbox = CGAL::bbox_3(points.begin(),points.end());

   const double min_x = bbox.xmin(), max_x = bbox.xmax(), min_y = bbox.ymin(),
        max_y = bbox.ymax(), min_z = bbox.zmin(), max_z = bbox.zmax();

    double length_x = max_x - min_x;
    double length_y = max_y - min_y;
    double length_z = max_z - min_z;
    double length = std::max(length_x, std::max(length_y, length_z));
    std::vector<Point_3> nored_points,centered_nored_points;
    for (auto& p : points)
    {
        nored_points.push_back(Point_3(((CGAL::to_double(p.x())-min_x)/length)*2-1,
            (CGAL::to_double(p.y()) - min_y) / length * 2 - 1,
           ( CGAL::to_double(p.z()) - min_z) / length * 2 - 1));
    }

    const auto bbox_2 = CGAL::bbox_3(nored_points.begin(), nored_points.end());
    const double x_c = (bbox_2.xmax() + bbox_2.xmin()) / 2;
    const double y_c = (bbox_2.ymax() + bbox_2.ymin()) / 2;
    const double z_c = (bbox_2.zmax() + bbox_2.zmin()) / 2;

    for (auto& p : nored_points)
    {
        centered_nored_points.push_back(Point_3(CGAL::to_double(p.x()) - x_c,
            CGAL::to_double(p.y()) - y_c,
            CGAL::to_double(p.z()) - z_c));
    }

    std::vector<it_V> point_to_it_V;
    for (int i = 0; i < centered_nored_points.size(); ++i)
        point_to_it_V.push_back(mesh.add_vertex(centered_nored_points[i]));
    for (auto face : indices)
    {
        std::vector<it_V>face_it_V;
        for (auto point : face)
            face_it_V.push_back(point_to_it_V[point]);
        mesh.add_face(face_it_V[0], face_it_V[1], face_it_V[2]);
    }
    //--------------------ik_mesh
    ik_mesh = mesh;

    //for (int i = 0; i < indices.size(); ++i)
    //{
    //    std::vector<unsigned int> link_face;
    //    for (int j = 0; j < indices.size(); ++j)
    //    {
    //        int same_point = 0;
    //        if (indices[j][0] == indices[i][0] || indices[j][1] == indices[i][0] || indices[j][2] == indices[i][0])
    //            ++same_point;
    //        if (indices[j][0] == indices[i][1] || indices[j][1] == indices[i][1] || indices[j][2] == indices[i][1])
    //            ++same_point;
    //        if (indices[j][0] == indices[i][2] || indices[j][1] == indices[i][2] || indices[j][2] == indices[i][2])
    //            ++same_point;
    //        if (same_point == 2)
    //            link_face.push_back(j);
    //    }
    //    face_link_matrix.push_back(link_face);
    //}

}
void Runner::handle_result_()
{
    selected_vertex.clear();
    selected_edge.clear();
    selected_face.clear();
    for (int i = 1; i < render_mesh.point_V_C_N.size(); i += 3)render_mesh.point_V_C_N[i] = point_inited_color;
    for (int i = 1; i < render_mesh.line_V_C_N.size(); i += 3)render_mesh.line_V_C_N[i] = line_inited_color;
    for (int i = 1; i < render_mesh.tri_V_C_N.size(); i += 3)render_mesh.tri_V_C_N[i] = tri_inited_color;
    for (auto simplex : result_) {
        if (simplex.level == 1)
            selected_vertex.push_back(CGAL::SM_Vertex_index(simplex.id));
        else if (simplex.level == 2) 
            selected_edge.push_back(CGAL::SM_Edge_index(simplex.id));
        else if (simplex.level == 3) 
            selected_face.push_back(CGAL::SM_Face_index(simplex.id));
    }
    for (auto t : selected_vertex) {
        render_mesh.point_V_C_N[t.idx()*3+1] = selected_point_color;
    }
    for (auto t : selected_edge) {
        render_mesh.line_V_C_N[t.idx() * 2 * 3 + 1] = selected_line_color;
        render_mesh.line_V_C_N[t.idx() * 2 * 3 + 4] = selected_line_color;
    }
    for (auto t : selected_face) {
        render_mesh.tri_V_C_N[t.idx() * 3 * 3 + 1] = selected_tri_color;
        render_mesh.tri_V_C_N[t.idx() * 3 * 3 + 4] = selected_tri_color;
        render_mesh.tri_V_C_N[t.idx() * 3 * 3 + 7] = selected_tri_color;
    }

}

void Runner::star()
{
    get_input();
    result_.assign(input_.begin(), input_.end());
    for (auto e_v : E_V_matrix) {
        bool is_simplex1_selected = false;
        for (auto temp : result_) {
            if (temp.level != 1)continue;
            if (temp.id == e_v.second) {
                is_simplex1_selected = true;
                break;
            }
        }
        if (is_simplex1_selected) {
            result_.push_back(simplex(e_v.first, 2));
        }
    }
    for (auto f_e : F_E_matrix) {
        bool is_simplex2_selected = false;
        for (auto temp : result_) {
            if (temp.level != 2)continue;
            if (temp.id == f_e.second) {
                is_simplex2_selected = true;
                break;
            }
        }
        if (is_simplex2_selected) {
            result_.push_back(simplex(f_e.first, 3));
        }
    }
    handle_result_(); 
    clear_result();
}
void Runner::closure()
{
    get_input();
    result_.assign(input_.begin(), input_.end());
    for (auto t : input_) {
        if (t.level == 3) {
            auto face_id = CGAL::SM_Face_index(t.id);
            for (auto v : mesh.vertices_around_face(mesh.halfedge(face_id))) {
                result_.push_back(simplex(v.idx(), 1));
            }
            for (auto e : mesh.halfedges_around_face(mesh.halfedge(face_id))) {
                auto edge = mesh.edge(e);
                result_.push_back(simplex(edge.idx(), 2));
            }
        }
        if (t.level == 2) {
            auto edge_id = CGAL::SM_Edge_index(t.id);
            auto v1 = mesh.vertex(edge_id, 0);
            auto v2 = mesh.vertex(edge_id, 1);
            result_.push_back(simplex(v1.idx(), 1));
            result_.push_back(simplex(v2.idx(), 1));
        }
    }
    handle_result_();
    clear_result();
}
void Runner::link()//会去掉一些选择，所以我们最好在每次操作之前都把我们的选择给清空，在这三个操作里面再重新加上.
{
    get_input();
    std::vector<simplex> init_input;
    init_input.assign(input_.begin(),input_.end());
    star();
    closure();
    std::vector<bool> cl_st_selected_v;
    std::vector<bool> cl_st_selected_e;
    std::vector<bool> cl_st_selected_f;
    build_bool_vec();
    cl_st_selected_v.assign(vertices_vec.begin(),vertices_vec.end());
    cl_st_selected_e.assign(edges_vec.begin(), edges_vec.end());
    cl_st_selected_f.assign(faces_vec.begin(), faces_vec.end());
    selected_vertex.clear();
    selected_edge.clear();
    selected_face.clear();
    for (auto t : init_input) {
        if (t.level == 1) selected_vertex.push_back(CGAL::SM_Vertex_index(t.id));
        if (t.level == 2)selected_edge.push_back(CGAL::SM_Edge_index(t.id));
        if (t.level == 3)selected_face.push_back(CGAL::SM_Face_index(t.id));
    }
    closure();
    star();
    build_bool_vec();
    for (int i = 0; i < cl_st_selected_v.size(); ++i) {
        if (vertices_vec[i])cl_st_selected_v[i] = 0;
    }
    for (int i = 0; i < cl_st_selected_e.size(); ++i) {
        if (edges_vec[i])cl_st_selected_e[i] = 0;
    }
    for (int i = 0; i < cl_st_selected_f.size(); ++i) {
        if (faces_vec[i])cl_st_selected_f[i] = 0;
    }
    for (int i = 0; i < cl_st_selected_v.size(); ++i) {
        if (cl_st_selected_v[i])result_.push_back(simplex(i, 1));
    }
    for (int i = 0; i < cl_st_selected_e.size(); ++i) {
        if (cl_st_selected_e[i])result_.push_back(simplex(i, 2));
    }
    for (int i = 0; i < cl_st_selected_f.size(); ++i) {
        if (cl_st_selected_f[i])result_.push_back(simplex(i, 3));
    }
    handle_result_();
    clear_result();
}


void Runner::select_reset()
{
    transform_ik(mesh,render_mesh);
    selected_face.clear();
    selected_edge.clear();
    selected_vertex.clear();
}
bool Runner::is_sim_complex()
{
    //------------------method1-----------------------------------------------------
    build_bool_vec();
    for (int i = 0; i < faces_vec.size(); ++i) {
        if (faces_vec[i]) {
            auto f_id = CGAL::SM_Face_index(i);
            for (auto hedge : mesh.halfedges_around_face(mesh.halfedge(f_id))) {
                auto edge = mesh.edge(hedge);
                if (!edges_vec[edge.idx()])return false;
            }
            for (auto vertex : mesh.vertices_around_face(mesh.halfedge(f_id))) {
                if (!vertices_vec[vertex.idx()])return false;
            }
        }
    }
    for (int i = 0; i < edges_vec.size(); ++i) {
        if (!edges_vec[i])continue;
        auto e_id = CGAL::SM_Edge_index(i);
        auto v1 = mesh.vertex(e_id, 0);
        auto v2 = mesh.vertex(e_id, 1);
        if (!vertices_vec[v1.idx()] || !vertices_vec[v2.idx()])return false;
    }
    return true;
    //---------------closure of complex is itself-----可以增加一个不会改变selected的方式
    //build_bool_vec();
    //std::vector<bool> v_vec, e_vec, f_vec;
    //v_vec.assign(vertices_vec.begin(), vertices_vec.end());
    //e_vec.assign(edges_vec.begin(), edges_vec.end());
    //f_vec.assign(faces_vec.begin(), faces_vec.end());
    //closure();
    //build_bool_vec();
    //for (int i = 0; i < vertices_vec.size(); ++i)if (v_vec[i] != vertices_vec[i])return false;
    //for (int i = 0; i < edges_vec.size(); ++i)if (e_vec[i] != edges_vec[i])return false;
    //for (int i = 0; i < faces_vec.size(); ++i)if (f_vec[i] != faces_vec[i])return false;
    //return true;

}
int Runner::is_pure_complex()
{
    build_bool_vec();
    if (selected_face.size()) {
        for (int i = 0; i < faces_vec.size(); ++i) {
            if (faces_vec[i]) {
                auto f_id = CGAL::SM_Face_index(i);
                for (auto hedge : mesh.halfedges_around_face(mesh.halfedge(f_id))) {
                    auto edge = mesh.edge(hedge);
                    if (!edges_vec[edge.idx()])return -1;
                }
                for (auto vertex : mesh.vertices_around_face(mesh.halfedge(f_id))) {
                    if (!vertices_vec[vertex.idx()])return -1;
                }
            }
        }
        return 3;
    }
    if (selected_edge.size()) {
        for (int i = 0; i < edges_vec.size(); ++i) {
            if (!edges_vec[i])continue;
            auto e_id = CGAL::SM_Edge_index(i);
            auto v1 = mesh.vertex(e_id, 0);
            auto v2 = mesh.vertex(e_id, 1);
            if (!vertices_vec[v1.idx()] || !vertices_vec[v2.idx()])return -1;
        }
        return 2;
    }
    if (selected_vertex.size())return 1;
    return 0;
}
void Runner::boundary()
{
    k_pure_complex = is_pure_complex();
    if (k_pure_complex == -1)return;
    std::vector<int> int_vertex_vec(mesh.number_of_vertices()),
        int_edge_vec(mesh.number_of_edges());
    for (int i = 0; i < int_vertex_vec.size(); ++i)int_vertex_vec[i] = 0;
    for (int i = 0; i < int_edge_vec.size(); ++i)int_edge_vec[i] = 0;
    if (k_pure_complex == 3) {
        for (auto face : selected_face) {
            for (auto h_edge : mesh.halfedges_around_face(mesh.halfedge(face))) {
                auto edge = mesh.edge(h_edge);
                int_edge_vec[edge.idx()]++;
                auto v1 = mesh.vertex(edge, 0);
                auto v2 = mesh.vertex(edge, 1);
                int_vertex_vec[v1.idx()]++;
                int_vertex_vec[v2.idx()]++;
            }
        }
    }
    if (k_pure_complex == 2) {
        for (auto edge : selected_edge) {
            auto v1 = mesh.vertex(edge, 0);
            auto v2 = mesh.vertex(edge, 1);
            int_vertex_vec[v1.idx()]++;
            int_vertex_vec[v2.idx()]++;
        }
    }

    for (int i = 0; i < int_vertex_vec.size(); ++i)
        if (int_vertex_vec[i] == 1)
            result_.push_back(simplex(i, 1));

    for (int i = 0; i < int_edge_vec.size(); ++i)
    {
        if (int_edge_vec[i] == 2)
            result_.push_back(simplex(i, 2));
    }

    handle_result_();
    closure();

}
void Runner::cur_call_back(GLFWwindow* window, double xpos, double ypos)
{
    if (on_select)
    {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
            (*instance).select(xpos, ypos);
    }
    else {
        if (ImGui::GetIO().WantCaptureMouse)
            return;
        int w, h;
        glfwGetWindowSize(window, &w, &h);

        glm::vec temp = glm::vec2(xpos, ypos);
        temp = transform_mouse(temp, w, h);

        if (camera.firstMouse)
        {
            camera.lastX = temp.x;
            camera.lastY = temp.y;
            camera.firstMouse = false;
        }

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
        {
            camera.lastX = temp.x;
            camera.lastY = temp.y;
            return;
        }

        camera.rotate(transform_mouse(glm::vec2(xpos, ypos), w, h));

        camera.lastX = temp.x;
        camera.lastY = temp.y;
    }
}
void processInput(GLFWwindow* window)//to do
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = static_cast<float>(2.5 * deltaTime);
    //if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    //    camera.pan( cameraSpeed * glm::vec2(1,0));
    //if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    //    camera.eye() -= cameraSpeed * camera.dir();
    //if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    //    camera.eye() -= glm::normalize(glm::cross(camera.dir(), camera.up())) * cameraSpeed;
    //if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    //    camera.eye() += glm::normalize(glm::cross(camera.dir(), camera.up())) * cameraSpeed;
}
void Runner::render_obj(Shader& shader)//if some change happened to obj ,call this function
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    shader.use();

//glenale use before render 
    glBindVertexArray(tri_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, tri_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3)*render_mesh.tri_V_C_N.size(), render_mesh.tri_V_C_N.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tri_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*render_mesh.idxs.size(), render_mesh.idxs.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(line_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, line_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh.line_V_C_N.size(), render_mesh.line_V_C_N.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, line_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh.idxs.size(), render_mesh.idxs.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
    glEnableVertexAttribArray(2);

  

    glBindVertexArray(point_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, point_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh.point_V_C_N.size(), render_mesh.point_V_C_N.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, point_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh.idxs.size(), render_mesh.idxs.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Vec3), (void*)(2 * sizeof(Vec3)));
    glEnableVertexAttribArray(2);

    return;
}

void Runner::render_map(Shader& shader)//if some change happened to ... call this
{//use a new frame buffer to store idx of the tri,and then 
 
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    unsigned int textureColorbuffer;
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
    // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << "\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenVertexArrays(1, &map_VAO);
    glGenBuffers(1, &map_VBO);
    glGenBuffers(1, &map_EBO);

    glBindVertexArray(map_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, map_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3) * render_mesh.V_Idx.size(), render_mesh.V_Idx.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, map_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * render_mesh.idxs.size(), render_mesh.idxs.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(Vec3), (void*)(1 * sizeof(Vec3)));
    glEnableVertexAttribArray(1);

}
bool Runner::read_mesh(std::string path)
{
    std::string extension = path.substr(path.size() - 3);
    IK_Mesh temp_mesh;
    if (extension == "off") {
        if (CGAL::IO::read_OFF(path, temp_mesh))
        {
            std::vector<Point_3> points;
            std::vector<std::vector<unsigned int>> indices;
            for (auto v = temp_mesh.vertices_begin(); v != temp_mesh.vertices_end(); ++v)
                points.push_back(temp_mesh.point(*v));
            for (auto face : temp_mesh.faces())
            {
                std::vector<unsigned int> index;
                for (auto v : temp_mesh.vertices_around_face(temp_mesh.halfedge(face)))
                    index.push_back(v.idx());
                indices.push_back(index);
            }
            get_mesh(points, indices);
            return 1;
        }
        else return 0;
    }
    else if (extension == "ply") {

        if (CGAL::IO::read_PLY(path, temp_mesh))
        {
            std::vector<Point_3> points;
            std::vector<std::vector<unsigned int>> indices;
            for (auto v = temp_mesh.vertices_begin(); v != temp_mesh.vertices_end(); ++v)
                points.push_back(temp_mesh.point(*v));
            for (auto face : temp_mesh.faces())
            {
                std::vector<unsigned int> index;
                for (auto v : temp_mesh.vertices_around_face(temp_mesh.halfedge(face)))
                    index.push_back(v.idx());
                indices.push_back(index);
            }
            get_mesh(points, indices);
            return 1;
        }
        else return 0;
    }
    else if (extension == "obj") {
        std::vector<Point_3> points;
        std::vector<std::vector<unsigned int>> indices;

        if (CGAL::IO::read_OBJ(path, points, indices))
        {
            get_mesh(points, indices);
            return 1;
        }
        else return 0;
    }
    return 0;
}

//void Runner::transform(Mesh& mesh,mesh_for_Render& render_mesh)
//{
//    //to_sum_one_center_zero
//   
//    //--------get_mesh_for_render
//    render_mesh.tri_V_C_N.clear();
//    render_mesh.line_V_C_N.clear();
//    render_mesh.point_V_C_N.clear();
//    render_mesh.idxs.clear();
//    render_mesh.V_Idx.clear();
//    int idx = 0;
//    render_mesh.point_V_C_N.resize(mesh.number_of_vertices() * 3);
//    for (auto face : mesh.faces())
//    {
//        Vec3 points[10];
//        int id = 0;
//        for (auto vd : vertices_around_face(mesh.halfedge(face), mesh)) {
//            double x = CGAL::to_double(mesh.point(vd).x());
//            double y = CGAL::to_double(mesh.point(vd).y());
//            double z = CGAL::to_double(mesh.point(vd).z());
//            points[id++] = Vec3(x,y, z);
//        }
//        auto& map = mesh.add_property_map<it_F, Vec3>("normal").first;
//        map[face] = compute_normal(points[0], points[1], points[2]);
//
//        for (it_V vd : vertices_around_face(mesh.halfedge(face), mesh)) {
//            double x = CGAL::to_double(mesh.point(vd).x());
//            double y = CGAL::to_double(mesh.point(vd).y());
//            double z = CGAL::to_double(mesh.point(vd).z());
//            render_mesh.tri_V_C_N.push_back(Vec3(x,y,z));
//
//            render_mesh.tri_V_C_N.push_back(tri_inited_color);
//            render_mesh.tri_V_C_N.push_back(mesh.property_map<it_F, Vec3>("normal").first[face]);
//            render_mesh.idxs.push_back(idx++);//face_idx/3 = face_id
//
//            //for point/vertex render_map
//            render_mesh.V_Idx.push_back(Vec3(x, y, z));
//            render_mesh.V_Idx.push_back(transform_idx(face.idx()+1));// readpixel will return face_id =1 
//
//        }
//    }
//
//    for (auto edge : mesh.edges()) {
//        auto v1 =mesh.vertex(edge, 0);
//        auto v2 = mesh.vertex(edge, 1);
//        auto face = mesh.face(edge.halfedge());
//        double x1 = CGAL::to_double(mesh.point(v1).x());
//        double y1 = CGAL::to_double(mesh.point(v1).y());
//        double z1 = CGAL::to_double(mesh.point(v1).z());
//        double x2 = CGAL::to_double(mesh.point(v2).x());
//        double y2 = CGAL::to_double(mesh.point(v2).y());
//        double z2 = CGAL::to_double(mesh.point(v2).z());
//        render_mesh.line_V_C_N.push_back(Vec3(x1,y1,z1));
//        render_mesh.line_V_C_N.push_back(line_inited_color);
//        render_mesh.line_V_C_N.push_back(mesh.property_map<it_F, Vec3>("normal").first[face]);
//        render_mesh.line_V_C_N.push_back(Vec3(x2,y2,z2));
//        render_mesh.line_V_C_N.push_back(line_inited_color);
//        render_mesh.line_V_C_N.push_back(mesh.property_map<it_F, Vec3>("normal").first[face]);
//
//        render_mesh.point_V_C_N[v1.idx() * 3] = (Vec3(x1,y1,z1));
//        render_mesh.point_V_C_N[v1.idx() * 3 + 1] = (point_inited_color);
//        render_mesh.point_V_C_N[v1.idx() * 3 + 2] = (mesh.property_map<it_F, Vec3>("normal").first[face]);
//    }
//}

void Runner::transform_ik(IK_Mesh& ik_mesh, mesh_for_Render& render_mesh)
{
    render_mesh.tri_V_C_N.clear();
    render_mesh.line_V_C_N.clear();
    render_mesh.point_V_C_N.clear();
    render_mesh.idxs.clear();
    render_mesh.V_Idx.clear();
    int idx = 0;
    render_mesh.point_V_C_N.resize(ik_mesh.number_of_vertices() * 3);
    for (auto face : ik_mesh.faces())
    {
        Vec3 points[10] = { Vec3(0, 0, 0) };
        int id = 0;
        for (auto vd : vertices_around_face(ik_mesh.halfedge(face), ik_mesh)) {
            double x = CGAL::to_double(ik_mesh.point(vd).x());
            double y = CGAL::to_double(ik_mesh.point(vd).y());
            double z = CGAL::to_double(ik_mesh.point(vd).z());
            points[id++] = Vec3(x, y, z);
        }

        auto& map = ik_mesh.add_property_map<it_F, Vec3>("normal").first;
        map[face] = compute_normal(points[0], points[1], points[2]);

        for (it_V vd : vertices_around_face(ik_mesh.halfedge(face), ik_mesh)) {
            double x = CGAL::to_double(ik_mesh.point(vd).x());
            double y = CGAL::to_double(ik_mesh.point(vd).y());
            double z = CGAL::to_double(ik_mesh.point(vd).z());
            render_mesh.tri_V_C_N.push_back(Vec3(x, y, z));

            render_mesh.tri_V_C_N.push_back(tri_inited_color);
            render_mesh.tri_V_C_N.push_back(ik_mesh.property_map<it_F, Vec3>("normal").first[face]);
            render_mesh.idxs.push_back(idx++);//face_idx/3 = face_id

            //for point/vertex render_map
            render_mesh.V_Idx.push_back(Vec3(x, y, z));
            render_mesh.V_Idx.push_back(transform_idx(face.idx() + 1));// readpixel will return face_id =1 

        }
    }
    /*puts("face_ok");
    for (auto edge : ik_mesh.edges()) {
        auto v1 = ik_mesh.vertex(edge, 0);
        auto v2 = ik_mesh.vertex(edge, 1);
        auto face = ik_mesh.face(edge.halfedge());
        double x1 = CGAL::to_double(ik_mesh.point(v1).x());
        double y1 = CGAL::to_double(ik_mesh.point(v1).y());
        double z1 = CGAL::to_double(ik_mesh.point(v1).z());
        double x2 = CGAL::to_double(ik_mesh.point(v2).x());
        double y2 = CGAL::to_double(ik_mesh.point(v2).y());
        double z2 = CGAL::to_double(ik_mesh.point(v2).z());
        render_mesh.line_V_C_N.push_back(Vec3(x1, y1, z1));
        render_mesh.line_V_C_N.push_back(line_inited_color);
        render_mesh.line_V_C_N.push_back(ik_mesh.property_map<it_F, Vec3>("normal").first[face]);
        render_mesh.line_V_C_N.push_back(Vec3(x2, y2, z2));
        render_mesh.line_V_C_N.push_back(line_inited_color);
        render_mesh.line_V_C_N.push_back(ik_mesh.property_map<it_F, Vec3>("normal").first[face]);

        render_mesh.point_V_C_N[v1.idx() * 3] = (Vec3(x1, y1, z1));
        render_mesh.point_V_C_N[v1.idx() * 3 + 1] = (point_inited_color);
        render_mesh.point_V_C_N[v1.idx() * 3 + 2] = (ik_mesh.property_map<it_F, Vec3>("normal").first[face]);
    }*/
}
void Runner::transform_polyhedron(Polyhedron& mesh, mesh_for_Render& render_mesh)
{
    render_mesh.tri_V_C_N.clear();
    render_mesh.line_V_C_N.clear();
    render_mesh.point_V_C_N.clear();
    render_mesh.idxs.clear();
    render_mesh.V_Idx.clear();
    int idx = 0;
    render_mesh.point_V_C_N.resize(mesh.points().size() * 3);
    for (auto face_handle = mesh.facets_begin(); face_handle != mesh.facets_end(); ++face_handle)
    {
        auto hf = face_handle->facet_begin();
        std::vector<Vec3> points;
        do
        {
            auto vd = hf->vertex()->point();
            double x = CGAL::to_double(vd.x());
            double y = CGAL::to_double(vd.y());
            double z = CGAL::to_double(vd.z());
            points.push_back(Vec3(x, y, z));
        } while (++hf != face_handle->facet_begin());      
        if (points.size() != 3)puts("not a tri");
        auto normal = compute_normal(points[0], points[1], points[2]);
        for (auto& p : points)
        {
            render_mesh.tri_V_C_N.push_back(p);
            render_mesh.tri_V_C_N.push_back(tri_inited_color);
            render_mesh.tri_V_C_N.push_back(normal);
            render_mesh.idxs.push_back(idx++);//face_idx/3 = face_id
        }
    }
}

void Runner::transform(libgl_mesh& mesh, mesh_for_Render& render_mesh)
{
    render_mesh.tri_V_C_N.clear();
    render_mesh.line_V_C_N.clear();
    render_mesh.point_V_C_N.clear();
    int idx = 0;
    render_mesh.point_V_C_N.resize(mesh.V.rows() * 3);

    for (int i = 0; i < mesh.F.rows(); ++i)
    {
        Vec3 points[10];

        points[0] =Vec3( CGAL::to_double(mesh.V(mesh.F(i, 0), 0)),
            CGAL::to_double(mesh.V(mesh.F(i, 0), 1)),
                CGAL::to_double(mesh.V(mesh.F(i, 0), 2)));

        points[1] = Vec3(CGAL::to_double(mesh.V(mesh.F(i, 1), 0)),
            CGAL::to_double(mesh.V(mesh.F(i, 1), 1)),
                CGAL::to_double(mesh.V(mesh.F(i, 1), 2)));

        points[2] = Vec3(CGAL::to_double(mesh.V(mesh.F(i, 2), 0)),
            CGAL::to_double(mesh.V(mesh.F(i, 2), 1)),
                CGAL::to_double(mesh.V(mesh.F(i, 2), 2)));

        render_mesh.tri_V_C_N.push_back(points[0]);
        render_mesh.tri_V_C_N.push_back(tri_inited_color);
        render_mesh.tri_V_C_N.push_back(compute_normal(points[0], points[1], points[2]));
        render_mesh.idxs.push_back(idx++);
        render_mesh.V_Idx.push_back(points[0]);
        render_mesh.V_Idx.push_back(transform_idx(i + 1));

        render_mesh.tri_V_C_N.push_back(points[1]);
        render_mesh.tri_V_C_N.push_back(tri_inited_color);
        render_mesh.tri_V_C_N.push_back(compute_normal(points[0], points[1], points[2]));
        render_mesh.idxs.push_back(idx++);
        render_mesh.V_Idx.push_back(points[1]);
        render_mesh.V_Idx.push_back(transform_idx(i + 1));

        render_mesh.tri_V_C_N.push_back(points[2]);
        render_mesh.tri_V_C_N.push_back(tri_inited_color);
        render_mesh.tri_V_C_N.push_back(compute_normal(points[0], points[1], points[2]));
        render_mesh.idxs.push_back(idx++);
        render_mesh.V_Idx.push_back(points[2]);
        render_mesh.V_Idx.push_back(transform_idx(i + 1));
    }
}
Vec3 Runner::transform_idx(unsigned int idx)// transform a idx to a Vec3(code this idx)
{
    std::vector<float>temp;
    while (idx)
    {
        temp.push_back(transform_map[idx % 255]);
        idx -= (idx % 255);
        idx /= 255;
    }
    if (temp.size() > 3) {
        printf("maybe some error in transform_idx or the model is too big\n");
    }
    while (temp.size() < 3)temp.push_back(0);
    return Vec3(temp[2], temp[1], temp[0]);
}
unsigned int Runner::reverse_transform_idx(Vec3 idx_code)
{
    unsigned int ans=(idx_code[2]+(idx_code[1]+idx_code[0]*255)*255)*255;
    return ans;
}
bool Runner::open_mesh()
{
    std::string read_path;
    read_path = window_select_file();
    if (read_path=="") {
        printf("fail to load read_path\n");
        return 0;
    }
    if (!read_mesh(read_path)) {
        printf("fail to read mesh from our path\n");
        return 0;
    }
    transform_ik(mesh,render_mesh);
    return 1;
}
void Runner::select(int xpos,int ypos)//let's get point directly
{
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    ypos = h - ypos;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    float rgb_arry[3];
    glReadPixels(xpos, ypos, 1, 1, GL_RGB, GL_FLOAT, rgb_arry);
    if (rgb_arry[0] < 0)return;
    unsigned int idx = reverse_transform_idx(Vec3(rgb_arry[0],
        rgb_arry[1], rgb_arry[2]));
    if (idx <= 0)return;

    size_t f_id = idx-1;
    if (select_simplex_level == 3&&render_mesh.tri_V_C_N[(f_id) * 3 * 3 + 1] != selected_tri_color){
        selected_face.push_back(CGAL::SM_Face_index(f_id));
        render_mesh.tri_V_C_N[(f_id) * 3 * 3 + 1] = selected_tri_color;
        render_mesh.tri_V_C_N[(f_id) * 3 * 3 + 4] = selected_tri_color;
        render_mesh.tri_V_C_N[(f_id) * 3 * 3 + 7] = selected_tri_color;
    }

    if (select_simplex_level == 2) {
        auto temp = CGAL::SM_Face_index(f_id);
        CGAL::SM_Edge_index selected_e;
        double min_dis = 1e9 + 10;
        for (auto t : mesh.halfedges_around_face(mesh.halfedge(temp)))
        {
            auto edge = mesh.edge(t);
            auto v1 = mesh.vertex(edge, 0);
            auto v2 = mesh.vertex(edge, 1);
            double x1 = CGAL::to_double(mesh.point(v1).x());
            double y1 = CGAL::to_double(mesh.point(v1).y());
            double z1 = CGAL::to_double(mesh.point(v1).z());
            double x2 = CGAL::to_double(mesh.point(v1).x());
            double y2 = CGAL::to_double(mesh.point(v1).y());
            double z2 = CGAL::to_double(mesh.point(v1).z());
            Vec2 p1= mvp_transform(Vec3(x1,y1,z1));
            Vec2 p2= mvp_transform(Vec3(x2,y2,z2));
            if (distance(p1, p2, Vec2(xpos, ypos)) < min_dis) {
                min_dis = distance(p1, p2, Vec2(xpos, ypos));
                selected_e = edge;
            }
        }
        if (render_mesh.line_V_C_N[selected_e.idx() * 6 + 1] != selected_line_color) {
            selected_edge.push_back(selected_e);
            render_mesh.line_V_C_N[selected_e.idx() * 6 + 1] = selected_line_color;
            render_mesh.line_V_C_N[selected_e.idx() * 6 + 4] = selected_line_color;
        }
    }

    if (select_simplex_level == 1) {
        auto temp = CGAL::SM_Face_index(f_id);
        CGAL::SM_Vertex_index selected_v;
        float min_dis = 1e5 + 10;
        for (auto v : mesh.vertices_around_face(mesh.halfedge(temp)))
        {
            
            Vec2 p = mvp_transform(Vec3(CGAL::to_double(mesh.point(v).x()), CGAL::to_double(mesh.point(v).y()),
                CGAL::to_double(mesh.point(v).z())));
            if (distance(p, Vec2(xpos, ypos)) < min_dis) {
                min_dis = distance(p, Vec2(xpos, ypos));
                selected_v = v;
            }
        }
        if (render_mesh.point_V_C_N[selected_v.idx() * 3 + 1] != selected_point_color)
        {
            selected_vertex.push_back(selected_v);
            render_mesh.point_V_C_N[selected_v.idx() * 3 + 1] = selected_point_color;
        }
    }

}
Vec2 Runner::mvp_transform(Vec3 point)
{
    auto temp = m_projection * m_view * m_model * glm::vec4(point.x, point.y, point.z, 1);
    int t1, t2;
    glfwGetWindowSize(window, &t1, &t2);
    temp.x /= temp.w;
    temp.y /= temp.w;
    temp.z /= temp.w;
    temp.x += 1, temp.y += 1;
    temp.x *= t1 / 2;
    temp.y *= t2 / 2;
    return Vec2(temp.x,temp.y);
}
float Runner::distance(Vec2 p1, Vec2 p2, Vec2 p3)
{
    Vec2 a = p2 - p1, b = p3 - p1;
    double crossp = a.x * b.y - a.y * b.x;
    return std::fabs(crossp/a.length());
}
float Runner::distance(Vec2 p1, Vec2 p2)
{
    return sqrt((p1.x-p2.x)* (p1.x - p2.x)+ (p1.y - p2.y) * (p1.y - p2.y));
}


int Runner::run(Runner& runner)
{
    SAVE_PATH = save_path;
    for (int i = 0; i < 256; ++i)transform_map[i] = (float)i / (float)255;
    instance = &runner;
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window); 
    glfwSetScrollCallback(window,my_ScrollCallback);
    glfwSetFramebufferSizeCallback(window,frame_buffer_callback);
    glfwSetCursorPosCallback(window,cur_call_back);
    glfwSetInputMode(window, GLFW_CURSOR,GLFW_CURSOR_NORMAL);
    glfwSwapInterval(1); // Enable vsync    
 
    if (!gladLoadGL())
    {
        std::cout << "glad init failure" << std::endl;
    }
//----------------------------------------------------------------------------------------------------------------------

    ResourceManager::LoadShader("src/gui/shader.vs",
        "src/gui/shader.fs", nullptr, "shader_one");
    shader = ResourceManager::GetShader("shader_one");

    ResourceManager::LoadShader("src/gui/map_shader.vs",
        "src/gui/map_shader.fs", nullptr, "shader_two");
    map_shader = ResourceManager::GetShader("shader_two");

    ResourceManager::LoadShader("src/gui/primitives.vs",
        "src/gui/primitives.fs", nullptr, "shader_three");
    Primitives_shader = ResourceManager::GetShader("shader_three");
//--------------------------------------------------------------------------------------------------
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsClassic();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    glGenVertexArrays(1, &mtri_VAO);
    glGenBuffers(1, &mtri_VBO);
    glGenBuffers(1, &mtri_EBO);
    glGenVertexArrays(1, &mline_VAO);
    glGenBuffers(1, &mline_VBO);
    glGenBuffers(1, &mline_EBO);
    glGenVertexArrays(1, &mpoint_VAO);
    glGenBuffers(1, &mpoint_VBO);
    glGenBuffers(1, &mpoint_EBO);
    glGenVertexArrays(1, &tri_VAO);
    glGenBuffers(1, &tri_VBO);
    glGenBuffers(1, &tri_EBO);
    glGenVertexArrays(1, &line_VAO);
    glGenBuffers(1, &line_VBO);
    glGenBuffers(1, &line_EBO);
    glGenVertexArrays(1, &point_VAO);
    glGenBuffers(1, &point_VBO);
    glGenBuffers(1, &point_EBO);

    //----check--------------------------
    int pri_id = 0;

    my_runner = this;
    //std::thread t(show);
    //t.detach();

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);
        glfwPollEvents();
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);//here because it's 0
        {
            glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
        ImGui::Begin("...");
        {
            ImGui::InputInt("input select_simplex_level", &select_simplex_level);
            if (ImGui::Button("open")) {//todo open & clear 
                if (!open_mesh()) {
                    return -1;
                }
                render_obj(shader);
                render_map(map_shader);
                got_render_mesh = true;
           /*     transform(*mesh_Vs.back(), render_mesh);*/
            }
            ImGui::SameLine();
            if (ImGui::Button("compute Visual Hull")) {
                //generate_Visual_hull_views();
                detect_planes();
                init_origin_measure(shader);
                ////get_matrices();
                set_M_v_bbox();
                get_carved_meshes();
                for (int i = 0; i < carve_meshes.size(); ++i)
                {
                    const std::string path = SAVE_PATH;
                    const std::string fullpath = path +"carve_mesh" + std::to_string(i) + ".off";
                    std::ofstream out(fullpath);
                    out << (*carve_meshes[i]);
                    Polyhedron p;
                    std::ifstream ifs1(fullpath);
                    ifs1 >> p;
                    bool p_intersecting = PMP::does_self_intersect<CGAL::Parallel_if_available_tag>(p, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, p)));
                    if (!p_intersecting)
                    {
                        Nef_polyhedron N_p(p);
                        Nef_carve_meshes.emplace_back(N_p);
                    }
                    //Polyhedron p;
                    //if (mesh_to_polyhedron(*carve_meshes[i], p))
                    //{
                    //    Nef_polyhedron N_p(p);
                    //    if (N_p.is_simple())
                    //        Nef_carve_meshes.emplace_back(N_p);
                    //}
                    //else
                    //{
                    //    std::cout<<"mesh_to_polyhedron_error\n";
                    //}
                }
                auto start = clock();
                get_primitives();
                auto end = clock();
                double endtime = (double)(end - start) / CLOCKS_PER_SEC;
                std::cout << "Total time:" << endtime << '\n';

                std::cout << "primitives_size:" << primitives.size() << '\n';
                for (int i = 0; i < primitives.size(); ++i)
                {
                    const std::string path = SAVE_PATH;
                    const std::string fullpath = path + std::to_string(i) + ".off";
                    std::ofstream out(fullpath);
                    out << (*primitives[i]);
                    Polyhedron p;
                    std::ifstream ifs1(fullpath);
                    ifs1 >> p;
                    bool p_intersecting = PMP::does_self_intersect<CGAL::Parallel_if_available_tag>(p, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, p)));
                    if (!p_intersecting)
                    {
                        Nef_polyhedron N_p(p);
                        primitives_polyhedron.emplace_back(N_p);
                    }
                }
                compute_M_v();
                const std::string path = SAVE_PATH;
                const std::string fullpath = path + "visual_hull_end" + ".off";
                std::ofstream out(fullpath);
                out << (*mesh_Vs.back());
                out.close();

                CGAL::convert_nef_polyhedron_to_polygon_mesh(Visual_hull, Visual_hull_mesh);
                const std::string fullpath_P = path + "visual_hull_end_polyhedra" + ".off";
                std::ofstream out_P(fullpath_P);
                out_P << Visual_hull_mesh;
                out_P.close();
                PMP::triangulate_faces(Visual_hull_mesh);
            }
            
            ImGui::SameLine();
            if (ImGui::Button("carve"))
            {
                carving();
                CGAL::convert_nef_polyhedron_to_polygon_mesh(Visual_hull, output_mesh);
                const std::string path = SAVE_PATH;
                const std::string fullpath_P = path + "carve_end_polyhedra" + ".off";
                std::ofstream out_P(fullpath_P);
                out_P << output_mesh;
                out_P.close();
                PMP::triangulate_faces(output_mesh);
                simplify(output_mesh);
                const std::string fullpath_ = path + "carve_end_mesh" + ".off";
                std::ofstream out_(fullpath_);
                out_ << output_mesh;
                out_.close();
            }

            ImGui::SameLine();
            if (ImGui::Button("show_next_primitives")) {
                //transform(*primitives[pri_id], render_mesh);

            }
            ImGui::SameLine();
            //if (ImGui::Button("select")){
            //    on_select = !on_select;//cursor callback use this bool
            //}
            //ImGui::SameLine();
            ImGui::SameLine();
            if (ImGui::Button("Input")) {
                transform_ik(mesh, render_mesh);
            }
            ImGui::SameLine();
            if (ImGui::Button("Visual hull")) {
                transform_ik(Visual_hull_mesh, render_mesh); 
            }
            ImGui::SameLine();
            if (ImGui::Button("Output")) {
                transform_ik(output_mesh, render_mesh);
            }
            
            //if (ImGui::Button("finish_select")) {
            //    select_reset();
            //}
            ImGui::SameLine();
            if (ImGui::Button("save")) {
                //save_result();
            }
            ImGui::SameLine();
            if (ImGui::Button("simplify an out put")) {
                simplify(ik_mesh);
                const std::string path = SAVE_PATH;
                const std::string fullpath = path + "simplfied" + ".off";
                std::ofstream out(fullpath);
                out << (ik_mesh);
            }
            ImGui::SameLine();
            if (ImGui::Button("check view_dir")) {
                auto temp_dir = measure_views[dir_id++].Dir;
                std::cout << temp_dir << '\n';
                auto unit_temp_dir = (Vec3(
                    CGAL::to_double(temp_dir.x())
                    , CGAL::to_double(temp_dir.y())
                    , CGAL::to_double(temp_dir.z())));
                auto temp_eye = Vec3(0, 0, 0) - unit_temp_dir;
                glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
                glm::mat4 projection = glm::ortho(-1.5f, 1.5f, -1.5f, 1.5f, -1.0f, 100.0f);
                glm::mat4 view = glm::lookAt(temp_eye, Vec3(0.f, 0.f, 0.f), Vec3(0.0f,1.0f,0.0f));
                glm::mat4 model = glm::mat4(1);
                m_model = model, m_view = view, m_projection = projection;
            }
            if (ImGui::TreeNode("Output"))
            {
                ImGui::Text("is_sim_complex: %d", _sim_complex);
                ImGui::Text("k_sim_complex: %d", k_pure_complex);
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Operators"))
            {
                ImGui::Bullet(); 
                if (ImGui::SmallButton("closure")) {                    
                    closure();
                }
                ImGui::Bullet(); 
                if (ImGui::SmallButton("star")) {
                    star();
                }
                ImGui::Bullet(); 
                if (ImGui::SmallButton("link")) {
                    link();
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("is_sim_complex")) {
                    if (is_sim_complex()) {
                        _sim_complex = 1;
                    }
                    else _sim_complex = 0;
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("is_pure_sim_complex")) {
                    k_pure_complex = is_pure_complex();
                }
                ImGui::Bullet();
                if (ImGui::SmallButton("boundary")) {
                    boundary();
                }
                ImGui::TreePop();
            }
        }
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
//------------------render_obj----------------------------------------------------------

        if (got_render_mesh) 
        {
            //render_tris
            {//show_primitives
                /*auto temp_dir = Visual_hull_views[pri_id].Dir;
                auto unit_temp_dir = Vec3(
                    CGAL::to_double(temp_dir.x()) / std::sqrt(CGAL::to_double(temp_dir.squared_length()))
                    , CGAL::to_double(temp_dir.y()) / std::sqrt(CGAL::to_double(temp_dir.squared_length()))
                    , CGAL::to_double(temp_dir.z()) / std::sqrt(CGAL::to_double(temp_dir.squared_length())));

                auto temp_eye = Vec3(0, 0, 0) - Vec3(
                    (double)3.0 / CGAL::to_double(temp_dir[2]) * unit_temp_dir.x,
                    (double)3.0 / CGAL::to_double(temp_dir[2]) * unit_temp_dir.y,
                    (double)3.0 / CGAL::to_double(temp_dir[2]) * unit_temp_dir.z);*/            
                //glm::mat4 view = glm::lookAt(temp_eye, unit_temp_dir, camera.up());
            }
            
            render_obj(shader);
            shader.use();
            shader.setInteger("u", 1);
            glEnable(GL_DEPTH_TEST);
            //glm::mat4 projection = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.1f, 100.0f);
            glm::mat4 projection = glm::perspective(glm::radians(fov), (float)1280 / (float)720, 0.1f, 100.0f);
            glm::mat4 view = glm::lookAt(camera.eye(), camera.dir(), camera.up());

            glm::mat4 model = glm::mat4(1);

            shader.setMat4("model",model);
            shader.setMat4("view", view);
            shader.setMat4("projection",projection);
            glBindVertexArray(tri_VAO);
            glDrawElements(GL_TRIANGLES, render_mesh.idxs.size(), GL_UNSIGNED_INT, 0);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(0);

            //render lines
// if different color for point,edge,tris,use three different vao,vbo;

            glEnable(GL_DEPTH_TEST);
            glEnable(GL_LINE_SMOOTH);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glLineWidth(1.5f);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            //glBindVertexArray(line_VAO);
            //glDrawElements(GL_TRIANGLES, (GLuint)render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBindVertexArray(line_VAO);
            glDrawArrays(GL_LINES, 0, render_mesh.line_V_C_N.size() / 3);
            glDisable(GL_LINE_SMOOTH);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glBindVertexArray(0);

            // render points

            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glPointSize(1.0);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            //glBindVertexArray(point_VAO);
            //glDrawElements(GL_TRIANGLES, render_mesh.idxs.size(), GL_UNSIGNED_INT, nullptr);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBindVertexArray(point_VAO);
            glDrawArrays(GL_POINTS, 0, render_mesh.point_V_C_N.size() / 3);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glBindVertexArray(0);

            // render for framebuffer map

            glBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
            glEnable(GL_DEPTH_TEST);
            //glEnable(GL_BLEND);
            //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glClearColor(-1, -1, -1, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            map_shader.use();
            map_shader.setMat4("model", model);
            map_shader.setMat4("view", view);
            map_shader.setMat4("projection", projection);
            glPointSize(10.0);
            glBindVertexArray(map_VAO);
            glDrawElements(GL_TRIANGLES, render_mesh.V_Idx.size()/2, GL_UNSIGNED_INT, nullptr);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
bool Runner::mesh_to_polyhedron(const IK_Mesh& mesh, Polyhedron& polyhedron)
{
    typedef typename Polyhedron::HalfedgeDS HalfedgeDS;
    // Postcondition: hds is a valid polyhedral surface.
    CGAL::Polyhedron_incremental_builder_3<HalfedgeDS> B(polyhedron.hds());
    B.begin_surface(mesh.number_of_vertices(), mesh.number_of_faces());
    typedef typename HalfedgeDS::Vertex   Vertex;
    typedef typename Vertex::Point Point;
    assert(V.cols() == 3 && "V must be #V by 3");
    for (auto&v:mesh.vertices())
    {
        B.add_vertex(mesh.point(v));
    }
    assert(F.cols() == 3 && "F must be #F by 3");
    for (auto& f:mesh.faces())
    {
        B.begin_facet();
        for (auto& v : mesh.vertices_around_face(mesh.halfedge(f)))
        {
            B.add_vertex_to_facet(v.idx());
        }
        B.end_facet();
    }
    if (B.error())
    {
        B.rollback();
        return false;
    }
    B.end_surface();
    return polyhedron.is_valid();
}

