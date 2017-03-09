#include <isam/isam.h>

#include <iostream>
#include <vector>

#include <lcmtypes/vs_object_collection_t.h>
#include <lcmtypes/vs_point3d_list_collection_t.h>
#include <bot_core/bot_core.h>
#include <pronto_utils/pronto_math.hpp>

// boost
#include <boost/algorithm/string.hpp>

#include <ConciseArgs>

using namespace std;
using namespace isam;
using namespace Eigen;

typedef std::list<isam::Node*> nodes_t;

// for each time step, we have some new nodes
std::vector<nodes_t> _nodes;

struct CommandLineConfig {
  std::string fname;
  std::string cloud_directory;
};

/**
 * Utility class to map non-continuous indices to continuous based
 * on time of appearance
*/
class IndexMapper {
  int next_i;
  std::map<int, int> _map_index;
  int create_if_needed(int i, bool& created) {
    return _map_index[i];
  }
public:
  IndexMapper() : next_i(0) {}

  /*
   * Provide translated index - entry must already exist.
   */
  int operator[](int i) {
    std::map<int, int>::iterator it = _map_index.find(i);
    require(it!=_map_index.end(), "Loader::IndexMapper::operator[]: object does not exist");
    return _map_index[i];
  }

  /*
   * Add new translation if it doesn't exist yet.
   * @return True if new entry was created.
   */
  bool add(int i) {
    bool added;
    std::map<int, int>::iterator it = _map_index.find(i);
    if (it==_map_index.end()) {
      // if entry does not exist, create one with next ID
      _map_index[i] = next_i;
      next_i++;
      added = true;
    } else {
      added = false;
    }
    return added;
  }
};

lcm_t* m_lcm;
Slam slam;
vector < Pose2d_Node* > pg1_nodes;

IndexMapper _pose_mapper;

void sendCollection(vector < Pose2d_Node* > nodes,int id,string label)
{
  vs_object_collection_t objs;	
  size_t n = nodes.size();
  if (n > 1) {
    objs.id = id;
    stringstream oss;
    string mystr;
    oss << "coll" << id;
    mystr=oss.str();	  

    objs.name = (char*) label.c_str();
    objs.type = VS_OBJECT_COLLECTION_T_AXIS3D;
    objs.reset = false;
    objs.nobjects = n;
    vs_object_t poses[n];
    for (size_t i = 0; i < n; i++) {
      Pose2d_Node* node =  nodes[i];
      isam::Pose2d  pose;
      pose = (*node).value();

      // direct Eigen Approach: TODO: test and use
      //Matrix3f m;
      //m = AngleAxisf(angle1, Vector3f::UnitZ())
      // *  * AngleAxisf(angle2, Vector3f::UnitY())
      // *  * AngleAxisf(angle3, Vector3f::UnitZ());
      Eigen::Quaterniond quat = euler_to_quat(0,0, pose.t());

      poses[i].id = i;
      poses[i].x = pose.x();
      poses[i].y = pose.y() ;
      poses[i].z = 0;
      poses[i].qw = quat.w();
      poses[i].qx = quat.x();
      poses[i].qy = quat.y();
      poses[i].qz = quat.z();
    }
    objs.objects = poses;
    vs_object_collection_t_publish(m_lcm, "OBJECT_COLLECTION", &objs);
  }
}

void add_prior() {
  // _nodes.resize(1); // the size of nodes so far
  // _pose_mapper.add(0); // index of the prior
  Pose2d prior_origin(0., 0., 0.);
  Pose2d_Node* a0 = new Pose2d_Node();
  // _nodes[_pose_mapper[0]].push_back(a0);
  slam.add_node(a0);
  Pose2d_Factor* p_a0 = new Pose2d_Factor(a0, prior_origin, SqrtInformation(100. * eye(3)));
  slam.add_factor(p_a0);
  pg1_nodes.push_back(a0);
}

void add_odometry(unsigned int idx_x0, unsigned int idx_x1, const Pose2d& measurement, const Noise& noise) {
  // only add another node, if the current one does not exist
  Pose2d_Node* a1;
  if(idx_x1 >= pg1_nodes.size()) {
    a1 = new Pose2d_Node();
    slam.add_node(a1);  
    pg1_nodes.push_back(a1);
  }
  else {
    a1 = pg1_nodes.at(idx_x1);
  }

  Pose2d_Pose2d_Factor* o_a01 = new Pose2d_Pose2d_Factor(pg1_nodes.at(idx_x0), a1, measurement, noise);
  slam.add_factor(o_a01);
}

void parse_file(std::string filename) {
  std::string line;
  std::ifstream isam_trajectories(filename);

  if (isam_trajectories.good())
  {
    int line_num = 1;
    while (std::getline(isam_trajectories,line))
    {
      std::vector<std::string> strs;
      boost::split(strs, line, boost::is_any_of("\t "));

      if(strs.at(0).compare("ODOMETRY") == 0){
        unsigned int idx_x0 = std::stoi(strs.at(1)), idx_x1 = std::stoi(strs.at(2));
        double x = std::stod(strs.at(3)), y = std::stod(strs.at(4)), t = std::stod(strs.at(5)), ixx = std::stod(strs.at(7)), ixy = std::stod(strs.at(8)), ixt = std::stod(strs.at(9)), iyy = std::stod(strs.at(10)), iyt = std::stod(strs.at(11)), itt = std::stod(strs.at(12));

        // if it's the first measurement
        if(idx_x0 == 0) {
          add_prior();
        }

        MatrixXd sqrtinf(3,3);
        sqrtinf <<
          ixx, ixy, ixt,
          0.,  iyy, iyt,
          0.,   0., itt;
        Pose2d measurement(x,y,t);

        add_odometry(idx_x0, idx_x1, measurement, SqrtInformation(sqrtinf));
        sendCollection(pg1_nodes,10000000,"Map - loop, optimized");
      }
      // print and update every 150 steps
      if(line_num % 500 == 0) {
        // std::cout << "Step " << line_num << std::endl;
        slam.update();
      }
      ++line_num;
    }
    slam.update();
    slam.batch_optimization();
    std::cout << "\n";
    sendCollection(pg1_nodes,10000000,"Map - loop, optimized");
    isam_trajectories.close();
  }

  else std::cout << "Unable to open file" << std::endl; 

}


int main(int argc, char **argv) {

  CommandLineConfig cl_cfg;
  cl_cfg.fname = ""; // where's the scan at?
  cl_cfg.cloud_directory = ""; // where're the clouds at?

  ConciseArgs parser(argc, argv, "");
  parser.add(cl_cfg.fname, "f", "filename", "ISAM file to process");
  parser.add(cl_cfg.cloud_directory, "cd", "cloud_directory", "Directory with relative pointclouds to each pose");
  parser.parse();


  m_lcm = lcm_create(NULL);

  std::cout << "Starting isam_read_file_lidar" << std::endl;
  
  parse_file(cl_cfg.fname);

  std::cout << "Exiting isam_read_file_lidar" << std::endl;

  return 0;
}
