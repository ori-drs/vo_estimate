// Filter Chain to process Georges Square Husky Dataset (from 2016 06 07)
// and to optimize the results with iSAM
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

#include <string>
#include <iostream>
#include <fstream>

#include <lcm/lcm-cpp.hpp>
#include <lcmtypes/bot_core.hpp>

#include <pronto_utils/pronto_vis.hpp> // visualize pt clds
#include <pronto_utils/pronto_conversions_lcm.hpp>
#include <ConciseArgs>

#include <path_util/path_util.h>

using namespace std;

struct CommandLineConfig
{
  std::string in_log_fname;
  std::string input_channel;
  int output_period;
  bool output_2d;
};

std::ofstream isam_trajectory;
std::ofstream world_trajectory;

class App{
  public:
    App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_);
    
    ~App(){
    }

  private:
    const CommandLineConfig cl_cfg_;    

    boost::shared_ptr<lcm::LCM> lcm_recv_;
    boost::shared_ptr<lcm::LCM> lcm_pub_;

    void poseHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::pose_t* msg);

    Isometry3dTime previousPoseT_;

    int isam_counter_;
};    

App::App(boost::shared_ptr<lcm::LCM> &lcm_recv_, boost::shared_ptr<lcm::LCM> &lcm_pub_, const CommandLineConfig& cl_cfg_) : 
       lcm_recv_(lcm_recv_), lcm_pub_(lcm_pub_), cl_cfg_(cl_cfg_), 
       previousPoseT_(0,Eigen::Isometry3d::Identity())
{

  lcm_recv_->subscribe( cl_cfg_.input_channel, &App::poseHandler,this);

  isam_trajectory.open("husky_isam_trajectory.txt");//, ios::out | ios::app | ios::binary);
  world_trajectory.open("husky_world_trajectory.txt");//, ios::out | ios::app | ios::binary);
  isam_counter_ = 0;
  cout <<"Constructed\n";
}




// 0 will skip first 10 frames and print the 10th
// 1 will print the first frame
int counter =0; 

void App::poseHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::pose_t* msg){
  counter++;
  Isometry3dTime currentPoseT = pronto::getPoseAsIsometry3dTime(msg);

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
  std::cout << msg->utime << " " << deltaPoseT.pose.translation().transpose() << "\n";


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
  world_trajectory << msg->utime  << " " << (isam_counter_) << " "
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
  world_trajectory << msg->utime  << " " << (isam_counter_) << " "
                  << currentPoseT.pose.translation().x() << " " << currentPoseT.pose.translation().y() << " " << currentPoseT.pose.translation().z() << " "
                  << current_quat.w() << " " << current_quat.x() << " " << current_quat.y() << " " << current_quat.z() << " "
                  << current_rpy[0] << " " << current_rpy[1] << " " << current_rpy[2] << "\n";
  world_trajectory.flush();
}


  previousPoseT_ = currentPoseT;
}



int main(int argc, char **argv){
  CommandLineConfig cl_cfg;
  cl_cfg.input_channel = "POSE_BODY";  
  cl_cfg.in_log_fname = "";
  cl_cfg.output_period = 1; // don't skip any. another option would be 10 - skip every 10
  cl_cfg.output_2d = false;
  double processing_rate = 1; // real time

  ConciseArgs parser(argc, argv, "simple-fusion");
  parser.add(cl_cfg.input_channel, "i", "input_channel", "input_channel - POSE_BODY typically");
  parser.add(cl_cfg.in_log_fname, "L", "in_log_fname", "Process this log file");
  parser.add(processing_rate, "pr", "processing_rate", "Processing Rate from a log [0=ASAP, 1=realtime]");  
  parser.add(cl_cfg.output_period, "o", "output_period", "Period between constraints output");  
  parser.add(cl_cfg.output_2d, "d", "output_2d", "output 2d isam constraints");  
  parser.parse();
  
  //
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

  App app= App(lcm_recv, lcm_pub, cl_cfg);
  while(0 == lcm_recv->handle());
}
