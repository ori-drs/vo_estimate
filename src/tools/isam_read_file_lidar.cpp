#include <isam/isam.h>

#include <iostream>
#include <vector>

#include <lcmtypes/vs_object_collection_t.h>
#include <lcmtypes/vs_point3d_list_collection_t.h>
#include <bot_core/bot_core.h>
#include <pronto_utils/pronto_math.hpp>
#include <pronto_utils/pronto_vis.hpp> // visualize pt clds

// boost
#include <boost/algorithm/string.hpp>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// usleep
#include <unistd.h>

#include <ConciseArgs>

using namespace std;
using namespace isam;
using namespace Eigen;

struct CommandLineConfig {
  std::string fname;
  std::string cloud_directory;
};

lcm_t* m_lcm;
pronto_vis* pc_vis_;
Slam slam;
vector < Pose2d_Node* > pg1_nodes;
vector < pcl::PointCloud<pcl::PointXYZRGB>::Ptr > pcs;
std::vector<Isometry3dTime> posesT;
const int PCOFFSETID = 1000000;
const uint8_t RESET = 1;

// Send a list of poses/objects as OBJECT_COLLECTION
int obj_coll_id=1;
obj_cfg oconfig = obj_cfg(obj_coll_id,   "Poses"  ,5,1);

// Send the collection of point cloud lists
ptcld_cfg pconfig = ptcld_cfg(2,  "Clouds"     ,1,1, obj_coll_id,0, {0.2,0,0.2} );

void add_prior() {
  Pose2d prior_origin(0., 0., 0.);
  Pose2d_Node* a0 = new Pose2d_Node();
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

void add_cloud(unsigned int idx_x0, const std::string dirname) {
  stringstream cloud_loc;
  cloud_loc << dirname << "/" << idx_x0 << ".pcd";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (cloud_loc.str(), *cloud) == -1) //* load the file
  {
    std::cerr << "Couldn't read file " << cloud_loc << std::endl;
    return;
  }

  pconfig.reset = 0; // keep previous points
  pc_vis_->ptcld_to_lcm(pconfig, *cloud, idx_x0, idx_x0);
}

void add_pose(unsigned int idx_x0) {
  Eigen::Isometry3d poseI = Eigen::Isometry3d::Identity();
  Pose2d_Node* node =  pg1_nodes.at(idx_x0);
  isam::Pose2d  pose;
  pose = (*node).value();


  Eigen::Quaterniond quat = euler_to_quat(0,0, pose.t());

  poseI.translation().x() = pose.x();
  poseI.translation().y() = pose.y();
  poseI.translation().z() = 0;
  poseI.rotate(quat);

  Isometry3dTime poseT = Isometry3dTime ( idx_x0, poseI  );
  posesT.push_back( poseT );
  oconfig.reset = 0; // keep poses

  pc_vis_->pose_to_lcm(oconfig, poseT);
}

void update_poses() {
  // map backwards all the poses
  std::vector<Isometry3dTime> posesTupdated;
  for(size_t i = 0u; i < pg1_nodes.size(); ++i) {
    Isometry3dTime poseT = posesT.at(i);
    Pose2d_Node* node = pg1_nodes[i];
    isam::Pose2d pose;
    pose = (*node).value();

    Eigen::Quaterniond quat = euler_to_quat(0,0, pose.t());

    poseT.pose.translation().x() = pose.x();
    poseT.pose.translation().y() = pose.y();
    poseT.pose.translation().z() = 0;
    poseT.pose.rotate(quat);

    posesTupdated.push_back(poseT);
  }

  oconfig.reset = 1;

  // update all the poses
  pc_vis_->pose_collection_to_lcm(oconfig, posesTupdated);
  oconfig.reset = 0;
}

void parse_file(const std::string filename, const std::string dirname) {
  std::string line;
  std::ifstream isam_trajectories(filename);

  if (isam_trajectories.good())
  {
    int line_num = 1;
    while (std::getline(isam_trajectories,line))
    {
      std::vector<std::string> strs;
      boost::split(strs, line, boost::is_any_of("\t "));

      // ODOMETRY 25151 25152 -0.00111292 0.00035533 0.00116187  50 0 0 50 0 100
      if(strs.at(0).compare("ODOMETRY") == 0){
        unsigned int idx_x0 = std::stoi(strs.at(1)), idx_x1 = std::stoi(strs.at(2));
        double x = std::stod(strs.at(3)), y = std::stod(strs.at(4)), t = std::stod(strs.at(5)), ixx = std::stod(strs.at(7)), ixy = std::stod(strs.at(8)), ixt = std::stod(strs.at(9)), iyy = std::stod(strs.at(10)), iyt = std::stod(strs.at(11)), itt = std::stod(strs.at(12)); // there's an empty space as a 6th char

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
        if(dirname.compare("") != 0) add_cloud(idx_x0, dirname);
        add_pose(idx_x0);
      }
      // update every 500 steps
      if(line_num % 500 == 0) {
        slam.update();
      }
      ++line_num;
    }
    slam.update();
    slam.batch_optimization();
    std::cout << "\n";
    update_poses();
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
  pc_vis_ = new pronto_vis( m_lcm );

  std::cout << "Starting isam_read_file_lidar" << std::endl;
  
  parse_file(cl_cfg.fname, cl_cfg.cloud_directory);

  std::cout << "Exiting isam_read_file_lidar" << std::endl;

  return 0;
}
