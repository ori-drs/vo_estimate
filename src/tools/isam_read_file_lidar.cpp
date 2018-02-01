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

struct Odometry {
  unsigned int id;  // the line number
  std::string name;  // always ODOMETRY
  unsigned int ref;  // reference frame
  unsigned int source;  // source frame, sort by this!
  Pose2d measurement;  // x, y, theta
  Noise cov;  // covariance consisting of varx, cov x y, cov x t, var y, cov y t, var t

  // sorting function
  bool operator < (const Odometry& o) const {
      return (source < o.source);
  }
};

std::vector<Odometry> odometry;

struct CommandLineConfig {
  std::string fname;
  std::string cloud_directory;
};

lcm_t* m_lcm;
pronto_vis* pc_vis_;
Slam slam;
vector < Pose2d_Node* > pg1_nodes;
vector<std::pair<unsigned int, unsigned int>> node_mapping;
vector < pcl::PointCloud<pcl::PointXYZRGB>::Ptr > pcs;
// all poses before correction
std::vector<Isometry3dTime> posesT;
pcl::PCDWriter writer_;
std::string dirname = "";

// Send a list of poses/objects as OBJECT_COLLECTION
int obj_coll_id = 1000000;
obj_cfg oconfig = obj_cfg(obj_coll_id, "Poses", 5, 1);

void sendCollection(vector <Pose2d_Node*> nodes, int id, string label) {
  vs_object_collection_t objs;
  size_t n = nodes.size();
  if (n > 1) {
    objs.id = id;
    objs.name = (char*)label.c_str();
    objs.type = VS_OBJECT_COLLECTION_T_POSE3D;
    objs.reset = false;
    objs.nobjects = n;
    vs_object_t poses[n];
    for (size_t i = 0; i < n; i++) {
      Pose2d_Node* node =  nodes[i];
      isam::Pose2d  pose;
      pose = (*node).value();

      Eigen::Quaterniond quat = euler_to_quat(0, 0, pose.t());

      poses[i].id = i;
      poses[i].x = pose.x();
      poses[i].y = pose.y();
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

// the source is idx_x1
bool node_exists(unsigned int idx_x1) {
  for (std::vector<std::pair<unsigned int, unsigned int>>::iterator it =
    node_mapping.begin(); it != node_mapping.end(); ++it) {
    unsigned int source = it->second;

    if (source == idx_x1) {
      return true;
    }
  }

  return false;
}

void add_cloud(unsigned int idx_x1) {
  stringstream cloud_loc;
  cloud_loc << dirname << "/" << idx_x1 << ".pcd";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  // it could be the case that there is no such cloud for a certain pose - just skip
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (cloud_loc.str(), *cloud) == -1) //* load the file
  {
    std::cerr << std::endl << "Couldn't read file " << cloud_loc.str() << std::endl;
    // push an empty pointcloud to the vector of pointclouds
    // when extracting the maps, the poses id must correspond to the cloud id
    pcs.push_back(cloud);
    return;
  }

  // normalize the utime
  float utimeN = (float) idx_x1 / (float) odometry.size();

  // convert to rgb
  float rgb[3];
  jet_rgb(utimeN, rgb);

  // colour the pointcloud
  for (size_t i = 0u; i < cloud->points.size(); ++i) {
    cloud->points[i].r = rgb[0]*255;
    cloud->points[i].g = rgb[1]*255;
    cloud->points[i].b = rgb[2]*255;
  }

  // add to a vector of clouds
  pcs.push_back(cloud);

  // Send the collection of point cloud lists
  ptcld_cfg pconfig = ptcld_cfg(2000000, "Clouds", 1, 1, obj_coll_id, 1, {rgb[0], rgb[1], rgb[2]});

  pconfig.reset = 0;  // keep previous points
  pc_vis_->ptcld_to_lcm(pconfig, *cloud, idx_x1, idx_x1);
}

void add_pose(unsigned int idx_x1) {
  Eigen::Isometry3d poseI = Eigen::Isometry3d::Identity();
  Pose2d_Node* node = pg1_nodes.at(idx_x1);
  isam::Pose2d  pose;
  pose = (*node).value();

  Eigen::Quaterniond quat = euler_to_quat(0, 0, pose.t());

  poseI.translation().x() = pose.x();
  poseI.translation().y() = pose.y();
  poseI.translation().z() = 0;
  poseI.rotate(quat);

  Isometry3dTime poseT = Isometry3dTime(idx_x1, poseI);
  posesT.push_back(poseT);
  oconfig.reset = 0;  // keep poses

  pc_vis_->pose_to_lcm(oconfig, poseT);
}


void add_odometry(unsigned int idx_x0, unsigned int idx_x1, const Pose2d&
  measurement, const Noise& noise) {
  // only add another node, if the current one does not exist
  Pose2d_Node* a1;
  bool node_existed = true;
  if (!node_exists(idx_x1)) {
    node_existed = false;
    a1 = new Pose2d_Node();
    slam.add_node(a1);
    pg1_nodes.push_back(a1);
    node_mapping.push_back(std::make_pair(pg1_nodes.size()-1, idx_x1));
  } else {
    a1 = pg1_nodes.at(idx_x1);
  }

  Pose2d_Pose2d_Factor* o_a01 = new Pose2d_Pose2d_Factor(pg1_nodes.at(idx_x0), a1, measurement, noise);
  slam.add_factor(o_a01);

  // only display another node, if the current one does not exist
  if(!node_existed) {
    // only add a pose and cloud in LCM if they don't exist already
    // IMPORTANT - first add the pose, then the cloud
    add_pose(idx_x1);
    if (dirname.compare("") != 0) add_cloud(idx_x1);
  }
}

void add_prior() {

  Pose2d prior_origin(0., 0., 0.);
  Pose2d_Node* a0 = new Pose2d_Node();
  Pose2d_Factor* p_a0 = new Pose2d_Factor(a0, prior_origin, SqrtInformation(100. * eye(3)));

  slam.add_node(a0);
  slam.add_factor(p_a0);

  // save the node
  pg1_nodes.push_back(a0);
  // associate the node id with the id of the prior (always 0)
  node_mapping.push_back(std::make_pair(pg1_nodes.size()-1, 0));
  // add to LCM
  add_pose(0);
  add_cloud(0);
}

void parse_file(const std::string filename) {
  std::string line;
  std::ifstream isam_trajectories(filename);

  if (isam_trajectories.good()) {
    unsigned int line_num = 1;
    while (std::getline(isam_trajectories,line))
    {
      std::vector<std::string> strs;
      boost::split(strs, line, boost::is_any_of("\t "));

      // ODOMETRY 25151 25152 -0.00111292 0.00035533 0.00116187  50 0 0 50 0 100
      if(strs.at(0).compare("ODOMETRY") == 0){
        unsigned int idx_x0 = std::stoi(strs.at(1)), idx_x1 = std::stoi(strs.at(2));
        double x = std::stod(strs.at(3)), y = std::stod(strs.at(4)), t = std::stod(strs.at(5)), ixx = std::stod(strs.at(7)), ixy = std::stod(strs.at(8)), ixt = std::stod(strs.at(9)), iyy = std::stod(strs.at(10)), iyt = std::stod(strs.at(11)), itt = std::stod(strs.at(12)); // there's an empty space as a 6th char

        MatrixXd sqrtinf(3,3);
        sqrtinf <<
          ixx, ixy, ixt,
          0.,  iyy, iyt,
          0.,   0., itt;
        Pose2d measurement(x,y,t);

        // add to a vector of odometries
        odometry.push_back({line_num, "ODOMETRY", idx_x0, idx_x1, measurement, SqrtInformation(sqrtinf)});
      }
      ++line_num;
    }
    isam_trajectories.close();
  }

  else std::cout << "Unable to open file" << std::endl; 

}

void extract_maps(const std::vector<int> start, const std::vector<int> end) {
  if(start.size() != end.size()) std::cout << "Invalid loop parameters" << std::endl;
  // the nodes should be the same size as the pointclouds, as for each node there is a pointcloud associated with it
  // for each map loop
  for(size_t i = 0u; i<start.size(); ++i) {
    // create a new pointcloud
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    // get the clouds between start[i] and end[i] (thus accumulate)
    int duration = end[i] - start[i];
    std::cout << "Map " << i << " total poses:" << duration << std::endl;
    // for each pose/cloud
    for(size_t j = 0u; j<duration;j++) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
      // Isometry3dTime currentPoseT = posesTupdated.at(start[i]+j);

      // retrieve the updated pose
      Pose2d_Node* node = pg1_nodes[node_mapping[start[i]+j].first];
      isam::Pose2d  pose;
      pose = (*node).value();

      Eigen::Quaterniond quat = euler_to_quat(0, 0, pose.t());

      // compute the transformation between the cloud and world(local)
      Eigen::Affine3d T_world_curr_pose = Eigen::Affine3d::Identity();
      T_world_curr_pose.translation() << pose.x(), pose.y(), 0;

      T_world_curr_pose.rotate(quat);

      // std::cout << "Transform matrix:" << T_world_curr_pose.matrix() << std::endl;
      // transform the point cloud with the inverse of the transformation b/w world and the current pose

      pcl::transformPointCloud(*pcs.at(node_mapping[start[i]+j].second), *transformed_cloud, T_world_curr_pose);

      cloud += *transformed_cloud;
    }

    // at this point we have a concatinated "cloud"
    std::stringstream ss;
    ss << "map_" << i << ".pcd";
    writer_.write<pcl::PointXYZRGB>(ss.str(), cloud, false);
  }
}

void process_odometry() {

  // sort the odometry vector by source
  std::sort(odometry.begin(), odometry.end());

  for (size_t i = 0u; i<odometry.size(); ++i) {

    // std::cout << "processing i: " << i << std::endl;
    Odometry o = odometry.at(i);

    // if it's the first measurement
    if(i == 0) {
      add_prior();
    }

    add_odometry(o.ref, o.source, o.measurement, o.cov);

    slam.update();
    sendCollection(pg1_nodes, obj_coll_id, "Poses");
  }

  slam.update();
  slam.batch_optimization();
  sendCollection(pg1_nodes, obj_coll_id, "Poses");
  std::cout << "\n";
}


int main(int argc, char **argv) {

  CommandLineConfig cl_cfg;
  cl_cfg.fname = ""; // where's the scan at?
  cl_cfg.cloud_directory = ""; // where're the clouds at?

  ConciseArgs parser(argc, argv, "");
  parser.add(cl_cfg.fname, "f", "filename", "ISAM file to process");
  parser.add(cl_cfg.cloud_directory, "cd", "cloud_directory", "Directory with relative pointclouds to each pose");
  parser.parse();

  dirname = cl_cfg.cloud_directory;

  m_lcm = lcm_create(NULL);
  pc_vis_ = new pronto_vis( m_lcm );

  std::cout << "[iSAM] Starting isam_read_file_lidar" << std::endl;

  parse_file(cl_cfg.fname);

  std::cout << "[iSAM] Finished reading file." << std::endl;
  std::cout << "[iSAM] Processing odometry." << std::endl;

  process_odometry();

  std::cout << "[iSAM] Finished processing odometry." << std::endl;
  std::cout << "[iSAM] Extracting maps." << std::endl;

  // extract pointcloud maps
  std::vector<int> start;
  start.push_back(0);
  start.push_back(569);
  start.push_back(1122);

  std::vector<int> end;
  end.push_back(568);
  end.push_back(1120);
  end.push_back(1676);

  extract_maps(start, end);

  std::cout << "[iSAM] Finished extracting maps." << std::endl;
  std::cout << "Exiting isam_read_file_lidar" << std::endl;

  return 0;
}
