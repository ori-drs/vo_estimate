// Filter Chain to process Georges Square Husky Dataset (from 2016 06 07)
// and to optimize the results with iSAM
//
// Filter Chain 1 - From LCM input ===========================================
//
// 1. Generate a Motion Trajectory using Visual Odometry (with IMU to constrain gravity drift):
// se-simple-fusion -P husky/robot.cfg  -m 3   -p POSE_BODY_ON_IMU -L ~/logs/husky/2016-06-07-outdoor-experiment-with-MS-and-sick/lcmlog-2016-06-07.00-camera-10fps-three-loops-only -pr 0  
// ... interested in the pose of the camera at each iteration at 10Hz (not the 100Hz IMU version)
//
// 2. Run this process to output the input constraints that isam needs (i.e. relative odometry)
// in this case a downsampled trajectory in 3D.
// se-vo-slam -L input_odometry.lcmlog -pr 0 -o 10
// It is also possible to process in 3D without down sampling:
// se-vo-slam -L input_odometry.lcmlog -pr 0
//
// 2b. The ever so subtle next step is to append to that trajectory some loop closure edges. Six were sufficient for this experiment
//
// 3. The iSAM executable can process the output text file husky_isam_trajectory.txt to produce a smoothed trajectory
// The following works well for a very large 1200 trajectory in 2D
// isam -L husky_isam_trajectory.txt  -W output.txt
// The following works well for a very large 12,000 trajectory
// isam -L husky_isam_trajectory.txt  -d 100 -u 100 -W output.txt
// (the result is published to LCM collections)
//
//
// Filter Chain 2 - From text input ===========================================  
// 
// 1. Use vo (as above) to write an pose_trajectory.txt of this format:
// utime x y z qw qx qy qz roll pitch yaw
//
// 2. Read the text file using the mode implemented below (but commented out)
// This will output the same files as #2 above
// 
// 3. Use iSAM as in #3 above

#include <string>
#include <iostream>
#include <fstream>

#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot_core.hpp>
#include <lcmtypes/visualization.h> // VS_OBJECT_COLLECTION_T_AXIS3D

#include <pronto_utils/pronto_vis.hpp> // visualize pt clds
#include <pronto_utils/pronto_conversions_lcm.hpp>
#include <ConciseArgs>

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <cloud_accumulate/cloud_accumulate.hpp>

// dift filters
#include "diftFilters/filtering.hpp"

#include <path_util/path_util.h>

using namespace std;

struct CommandLineConfig
{
  std::string in_log_fname;
  std::string input_channel;
  int output_period;
  bool output_2d;
  std::string lidar_channel;
  bool save_pcs;
  bool filter_ground_points;
  float angleLeft;
  float angleRight;
  bool filter_range_points;
};

std::ofstream isam_trajectory;
std::ofstream world_trajectory;

const int PCOFFSETID = 1000000;
const uint8_t RESET = 1;
std::unique_ptr<dift::AbstractFilters> filters;

class App{
  public:
    App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_, const CloudAccumulateConfig& ca_cfg, const FilteringParams& filtering_params);
    
    ~App(){}

  private:
    BotParam* botparam_;
    BotFrames* botframes_;

    const CommandLineConfig cl_cfg_;

    // filters
    const FilteringParams filtering_params_;

    // cloud accumulate
    CloudAccumulate* accu_;
    CloudAccumulateConfig ca_cfg_;

    boost::shared_ptr<lcm::LCM> lcm_recv_;
    boost::shared_ptr<lcm::LCM> lcm_pub_;

    void poseHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const bot_core::pose_t* msg);
    void lidarHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const bot_core::planar_lidar_t* msg);

    void loadPoseTrajectory(std::string database_path, std::string fname);
    void convertPoseToIsam(Isometry3dTime currentPoseT);

    pronto_vis* pc_vis_;

    Isometry3dTime previousPoseT_;

    int isam_counter_;

    std::vector<Isometry3dTime> pose_trajectory_;

    boost::circular_buffer<std::shared_ptr<bot_core::planar_lidar_t>> messages_buffer_;

    pcl::PCDWriter writer_;

    std::string dirname_;
};    

App::App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_, const CloudAccumulateConfig& ca_cfg, const FilteringParams&filtering_params) : 
       lcm_recv_(lcm_recv_), lcm_pub_(lcm_pub_), cl_cfg_(cl_cfg_), 
       previousPoseT_(0,Eigen::Isometry3d::Identity()), ca_cfg_(ca_cfg), messages_buffer_(14), filtering_params_(filtering_params)
{
  do {
    botparam_ = bot_param_new_from_server(lcm_recv_->getUnderlyingLCM(), 0); // 1 means keep updated, 0 would ignore updates
  } while (botparam_ == NULL);

  botframes_= bot_frames_get_global(lcm_recv_->getUnderlyingLCM(), botparam_);

  // init accumulator
  accu_ = new CloudAccumulate(lcm_recv_, ca_cfg_, botparam_, botframes_);

  // init filters
  filters = std::unique_ptr<dift::AbstractFilters>(new dift::AbstractFilters(filtering_params_));

  // Populate buffer with NULL
  for(size_t i=0u;i<messages_buffer_.size();++i){
    messages_buffer_.push_back(NULL);
  }

  // create dir for pcs
  if(cl_cfg_.save_pcs) {
    dirname_ = "isam-clouds/";

    // check if directory exists - if so - delete it
    if(boost::filesystem::exists(dirname_)) {
      std::cout << "Directory " << dirname_ << " already exists. Re-creating it."<< std::endl;
      boost::filesystem::remove_all(dirname_);
    }

    if(boost::filesystem::create_directory(dirname_)) {
      std::cout << "Pointclouds will be stored at " << dirname_ << std::endl;
    }
    else {
      std::cout << "Couldn't create " << dirname_ << ". Will store pointclouds here." << std::endl;
      dirname_ = "";
    }


  }

  lcm_recv_->subscribe( cl_cfg_.input_channel, &App::poseHandler,this);
  lcm_recv_->subscribe( cl_cfg_.lidar_channel, &App::lidarHandler,this);

  isam_trajectory.open("husky_isam_trajectory.txt");//, ios::out | ios::app | ios::binary);
  world_trajectory.open("husky_world_trajectory.txt");//, ios::out | ios::app | ios::binary);
  isam_counter_ = 0;
  cout <<"Constructed\n";


  // Mode for Filter Chain 2 mode
  
  pc_vis_ = new pronto_vis( lcm_pub_->getUnderlyingLCM() ); // 4 triangle, 5 triad
  /*
  pc_vis_->obj_cfg_list.push_back( obj_cfg(4004,"Pose Trajectory",5,1) );
  std::string database_path = "/home/mfallon/logs/husky/2016-07-georges-square-west/";
  std::string fname = "husky_trajectory_2_experiment.txt";
  loadPoseTrajectory(database_path, fname);
  pc_vis_->pose_collection_to_lcm_from_list(4004, pose_trajectory_);
  for (size_t i = 0; i < pose_trajectory_.size() ; i++){
     convertPoseToIsam(pose_trajectory_[i] );
  }
  cout <<"Finished converting to iSAM format\n";
  exit(-1);
  */
}



void App::loadPoseTrajectory(std::string database_path, std::string fname){

  string full_filename =  database_path + fname;
  //printf( "About to read: %s - ",full_filename.c_str());
  string line0;
  std::ifstream this_file (full_filename.c_str());
  if (this_file.is_open()){

    getline (this_file,line0);
    //cout << line0 << " is first line\n";

    while ( this_file.good() ){
      string line;
      getline (this_file,line);
      if (line.size() > 4){

        int64_t utime;

        double p[3], q[4], r[3];
        sscanf(line.c_str(), "%ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&utime,
            &(p[0]), &(p[1]), &(p[2]), // position
            &(q[0]), &(q[1]), &(q[2]),  &(q[3]), // quat wxyz
            &(r[0]), &(r[1]), &(r[2])); // rpy (unused)

        Eigen::Isometry3d pose;
        pose.setIdentity();
        pose.translation() << p[0],p[1],p[2];
        Eigen::Quaterniond quat(q[0],q[1],q[2],q[3]);
        pose.rotate(quat);
        Isometry3dTime poseT = Isometry3dTime(utime, pose);

        //cout << utime << " | ";
        //cout << "p: " << p[0] << " " << p[1] << " " << p[2] << " | ";
        //cout << "q: " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << " | ";
        //cout << "r: " << r[0] << " " << r[1] << " " << r[2] << "\n";

        pose_trajectory_.push_back(poseT);
      }
    }
    this_file.close();
  } else{
    printf( "Unable to open trajectory file\n%s",fname.c_str());
    exit(-1);
    return;
  }
  cout << "Loaded pose trajectory of length " << pose_trajectory_.size() << "\n";
}

void App::lidarHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::planar_lidar_t* msg){
  if(cl_cfg_.save_pcs) {
    // Push into a queue and pull from front, to add some delay redundency
    // here we are using a 3 sample queue - so about 0.075sec of delay added
    std::shared_ptr<bot_core::planar_lidar_t> this_msg = std::shared_ptr<bot_core::planar_lidar_t>(new bot_core::planar_lidar_t(*msg));
    messages_buffer_.push_back(this_msg);

    // get the front one
    std::shared_ptr<bot_core::planar_lidar_t> processing_msg = messages_buffer_[0];
    if (processing_msg == NULL)
      return;

    if(cl_cfg_.filter_ground_points) {
      bot_core::planar_lidar_t filtered_points_msg;
      filters->filter(*processing_msg, filtered_points_msg);

      std::shared_ptr<bot_core::planar_lidar_t> filtered_points_msg_shrd = std::shared_ptr<bot_core::planar_lidar_t>(new bot_core::planar_lidar_t(filtered_points_msg));

      accu_->processLidar(filtered_points_msg_shrd);
    }
    else {
      accu_->processLidar(processing_msg);
    }
  }
}


void App::poseHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::pose_t* msg){
  Isometry3dTime currentPoseT = pronto::getPoseAsIsometry3dTime(msg);
  convertPoseToIsam(currentPoseT);
}

// 0 will skip first 10 frames and print the 10th
// 1 will print the first frame
int counter =0; 
void App::convertPoseToIsam(Isometry3dTime currentPoseT){
  counter++;

  if (previousPoseT_.utime == 0){
    std::cout << "initializing pose\n";
    previousPoseT_ = currentPoseT;
    return;
  }

  //std::cout << msg->utime << "\n";
  if (counter% cl_cfg_.output_period > 0){
    return;
  }

  // 1. Calculate the relative distance travelled since the last output:
  Eigen::Isometry3d deltaPose = previousPoseT_.pose.inverse() * currentPoseT.pose ;  
  Isometry3dTime deltaPoseT = Isometry3dTime(currentPoseT.utime, deltaPose);
  //bot_core::pose_t msg_out = pronto::getIsometry3dAsBotPose(deltaPoseT.pose, deltaPoseT.utime);
  //lcm_->publish(std::string("POSE_BODY_DELTA"),&msg_out);
  std::cout << currentPoseT.utime << " " << deltaPoseT.pose.translation().transpose() << "\n";


  // 2. Output the iSAM constraints (in either 2D or 3D) and a trajectory 
  isam_counter_++;
  if (cl_cfg_.output_2d){
    double delta_rpy[3];
    quat_to_euler( Eigen::Quaterniond( deltaPoseT.pose.rotation() ) , delta_rpy[0], delta_rpy[1], delta_rpy[2]);
    isam_trajectory << "ODOMETRY " << (isam_counter_-1) << " " << isam_counter_ << " " 
                    << deltaPoseT.pose.translation().x() << " "<< deltaPoseT.pose.translation().y() << " "
                    << delta_rpy[2] << " "
                    << " 50 0 0 50 0 100\n";
    isam_trajectory.flush();


    double current_rpy[3];
    quat_to_euler( Eigen::Quaterniond( currentPoseT.pose.rotation() ) , current_rpy[0], current_rpy[1], current_rpy[2]);
    world_trajectory << currentPoseT.utime  << " " << (isam_counter_) << " "
                    << currentPoseT.pose.translation().x() << " " << currentPoseT.pose.translation().y() << " "
                    << current_rpy[2] << "\n";
    world_trajectory.flush();
  }else{
    double delta_rpy[3];
    quat_to_euler( Eigen::Quaterniond( deltaPoseT.pose.rotation() ) , delta_rpy[0], delta_rpy[1], delta_rpy[2]);
    isam_trajectory << "EDGE3 " << (isam_counter_-1) << " " << isam_counter_ << " " 
                    << deltaPoseT.pose.translation().x() << " " << deltaPoseT.pose.translation().y() << " " << deltaPoseT.pose.translation().z() << " "
                    << delta_rpy[0] << " " << delta_rpy[1] << " " << delta_rpy[2] << " "
                    << " 10 0 0 0 0 0 10 0 0 0 0 10 0 0 0 100 0 0 100 0 25\n";
    //                << " xx xyxzr p y yy z r p y zz r p y rr  p y pp  y y"
    isam_trajectory.flush();



    double current_rpy[3];
    Eigen::Quaterniond current_quat = Eigen::Quaterniond( currentPoseT.pose.rotation() );
    quat_to_euler( current_quat  , current_rpy[0], current_rpy[1], current_rpy[2]);
    world_trajectory << currentPoseT.utime  << " " << (isam_counter_) << " "
                    << currentPoseT.pose.translation().x() << " " << currentPoseT.pose.translation().y() << " " << currentPoseT.pose.translation().z() << " "
                    << current_quat.w() << " " << current_quat.x() << " " << current_quat.y() << " " << current_quat.z() << " "
                    << current_rpy[0] << " " << current_rpy[1] << " " << current_rpy[2] << "\n";
    world_trajectory.flush();
  }

  // retrieve the accumulated point cloud up until now and store it relative to the current pose - isam_counter_
  if(cl_cfg_.save_pcs) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());

    pc_vis_->convertCloudProntoToPcl(*accu_->getCloud(), *cloud_xyzrgb);

    // because of filtering we could be left with no points in our pointcloud - then skip the saving
    if(cloud_xyzrgb->points.size() <= 0) {
      // clear the cloud
      accu_->clearCloud();
      // update the pose
      previousPoseT_ = currentPoseT;
      // stop execution
      return;
    }

    // now apply the transformation matrix that is world to the current pose inverse
    Eigen::Affine3d T_world_curr_pose = Eigen::Affine3d::Identity();
    T_world_curr_pose.translation() << currentPoseT.pose.translation().x(), currentPoseT.pose.translation().y(), currentPoseT.pose.translation().z();

    Eigen::Quaterniond current_quat = Eigen::Quaterniond( currentPoseT.pose.rotation() );
    T_world_curr_pose.rotate(current_quat);

    // std::cout << "Transform matrix:" << T_world_curr_pose.inverse().matrix() << std::endl;
    // transform the point cloud with the inverse of the transformation b/w world and the current pose
    pcl::transformPointCloud (*cloud_xyzrgb, *transformed_cloud, T_world_curr_pose.inverse());

    pronto::PointCloud* transformed_cloud_pronto (new pronto::PointCloud);
    pc_vis_->convertCloudPclToPronto(*transformed_cloud,*transformed_cloud_pronto);

    transformed_cloud->width = 1;
    transformed_cloud->height = transformed_cloud->points.size();

    // save the transformed point cloud
    std::stringstream ss;
    ss << dirname_ << (isam_counter_-1) << ".pcd";
    writer_.write<pcl::PointXYZRGB> (ss.str (), *transformed_cloud, false);


    
    // visualization
    // id, name, visualise as, reset
    std::stringstream ss_pose_name;
    ss_pose_name << "Pose " << (isam_counter_-1);
    obj_cfg ocfg = obj_cfg((isam_counter_-1), ss_pose_name.str(), VS_OBJECT_COLLECTION_T_AXIS3D, RESET);
    pc_vis_->pose_to_lcm(ocfg,currentPoseT);

    
    // id, name, visualsie as, reset, frame_id, (-1 - automatically asign, 0 - no color, 1 - use the rgb param), rgb
    std::stringstream ss_cl_name;
    ss_cl_name << "Cloud " << PCOFFSETID+(isam_counter_-1);
    ptcld_cfg pconfig = ptcld_cfg(PCOFFSETID+(isam_counter_-1),  ss_cl_name.str(),1,RESET, (isam_counter_-1),1,{0.2,0.2,0.2} );
    // 0
    pc_vis_->ptcld_to_lcm(pconfig, *transformed_cloud_pronto, currentPoseT.utime, currentPoseT.utime);
    
    // empty the accumulator
    accu_->clearCloud();
  }

  previousPoseT_ = currentPoseT;
}



int main(int argc, char **argv) {
  CommandLineConfig cl_cfg;
  cl_cfg.input_channel = "POSE_BODY";  
  cl_cfg.in_log_fname = "";
  cl_cfg.output_period = 1; // don't skip any. another option would be 10 - skip every 10
  cl_cfg.output_2d = false;
  cl_cfg.save_pcs = true;
  cl_cfg.lidar_channel = "SICK_SCAN";
  double processing_rate = 1; // real time
  cl_cfg.filter_ground_points = false;
  cl_cfg.filter_range_points = false;
  cl_cfg.angleLeft = -135; // left angle
  cl_cfg.angleRight = 135; // right angle


  CloudAccumulateConfig ca_cfg;
  ca_cfg.batch_size = 99999;
  ca_cfg.lidar_channel = cl_cfg.lidar_channel;
  ca_cfg.max_range = 999;
  ca_cfg.min_range = 0;

  ConciseArgs parser(argc, argv, "simple-fusion");
  parser.add(cl_cfg.input_channel, "i", "input_channel", "input_channel - POSE_BODY typically");
  parser.add(cl_cfg.in_log_fname, "L", "in_log_fname", "Process this log file");
  parser.add(processing_rate, "pr", "processing_rate", "Processing Rate from a log [0=ASAP, 1=realtime]");
  parser.add(cl_cfg.output_period, "o", "output_period", "Period between constraints output");
  parser.add(cl_cfg.output_2d, "d", "output_2d", "output 2d isam constraints");
  parser.add(cl_cfg.save_pcs, "s", "save_pointclouds", "saves scans relative to poses");
  parser.add(cl_cfg.lidar_channel, "lc", "lidar_channel", "which lidar channel to listen to");
  parser.add(cl_cfg.filter_ground_points, "fg", "filter_ground", "should I filter ground points?");
  parser.add(cl_cfg.angleLeft, "fgleft", "filter_ground_angle_left", "filter points below this angle on the left[deg]");
  parser.add(cl_cfg.angleRight, "fgright", "filter_ground_angle_right", "filter points below this angle on the right[deg]");
  parser.add(cl_cfg.filter_range_points, "fr", "filter_range", "should I filter points b/w certain range?");
  parser.add(ca_cfg.min_range, "frmin", "filter_range_min", "minimum distance from the lidar[m]");
  parser.add(ca_cfg.max_range, "frmax", "filter_range_max", "maximum distance from the lidar[m]");
  // parser.add(cl_cfg.filter_ground_points, "fg", "filter_ground", "should I filter ground points?");
  parser.parse();

  FilteringParams params;
  if(cl_cfg.filter_ground_points) {
    params.types.push_back("Ground");
    params.ground.angleLeft = cl_cfg.angleLeft;
    params.ground.angleRight = cl_cfg.angleRight;
  }

  // summary of params
  std::cout << "Running vo-slam-lidar with the following params:"           << std::endl;

  std::cout << "Input channel: "             << cl_cfg.input_channel        << std::endl;
  std::cout << "Log filename: "              << cl_cfg.in_log_fname         << std::endl;
  std::cout << "Downsampling: "              << cl_cfg.output_period        << std::endl;
  std::cout << "2D output: "                 << cl_cfg.output_2d            << std::endl;
  std::cout << "Save pointclouds: "          << cl_cfg.save_pcs             << std::endl;
  std::cout << "LIDAR channel: "             << cl_cfg.lidar_channel        << std::endl;
  std::cout << "Ground pts filter: "         << cl_cfg.filter_ground_points << std::endl;
  if(cl_cfg.filter_ground_points) {
  std::cout << "Ground Filter Left angle: "  << cl_cfg.angleLeft            << std::endl;
  std::cout << "Ground Filter Right angle: " << cl_cfg.angleRight           << std::endl;
  }
  std::cout << "Range pts filter: "          << cl_cfg.filter_range_points  << std::endl;
  if(cl_cfg.filter_range_points) {
  std::cout << "Range Filter min distance: " << ca_cfg.min_range            << std::endl;
  std::cout << "Range Filter max distance: " << ca_cfg.max_range            << std::endl;
  }
  
  bool running_from_log = !cl_cfg.in_log_fname.empty();
  boost::shared_ptr<lcm::LCM> lcm_recv;
  boost::shared_ptr<lcm::LCM> lcm_pub;
  if (running_from_log) {
    printf("running from log file: %s\n", cl_cfg.in_log_fname.c_str());
    std::stringstream lcmurl;
    lcmurl << "file://" << cl_cfg.in_log_fname << "?speed=" << processing_rate;
    lcm_recv = boost::shared_ptr<lcm::LCM>(new lcm::LCM(lcmurl.str()));
    if (!lcm_recv->good()) {
      fprintf(stderr, "Error couldn't load log file %s\n", lcmurl.str().c_str());
      exit(1);
    }
  }
  else {
    lcm_recv = boost::shared_ptr<lcm::LCM>(new lcm::LCM);
  }
  lcm_pub = boost::shared_ptr<lcm::LCM>(new lcm::LCM);


  App app= App(lcm_recv, lcm_pub, cl_cfg, ca_cfg, params);
  while(0 == lcm_recv->handle());
}
